from ET.et_iql import ETIQLModel, ETOfflineRLBaseModel
import torch.nn as nn
import torch
import os
from gen import constants
from unsupervised_baselines import utils
import unsupervised_baselines.utils as utils
from ET.et_model import FeatureFlat

class APTETIQLModel(ETIQLModel):
    
    def __init__(self, args, pad=0, seg=1):
        """
        apt IQL agent
        """
        super().__init__(args, pad, seg)
        self.encoder = FeatureFlat(input_shape=(512, 7, 7), output_size=1024)
        vocab_obj_path = os.path.join(constants.OBJ_CLS_VOCAB)
        vocab_obj = torch.load(vocab_obj_path)
        num_objects = len(vocab_obj)
        self.aug = utils.RandomShiftsAug(pad=1)
        self.hidden_dim = 1024
        self.icm = ICM(1024, num_objects, self.hidden_dim, 768)
        self.icm_scale = 1.0 
        self.knn_clip = 0.0
        self.knn_k = 12 # TODO: make a parameter
        self.knn_avg = True
        self.knn_rms = False
        
        # particle-based entropy
        rms = utils.RMS(args.device)
        self.pbe = utils.PBE(rms, self.knn_clip, self.knn_k, self.knn_avg, self.knn_rms, args.device)
        self.icm_opt = torch.optim.Adam(self.icm.parameters() + self.encoder, lr=1e-4)
        self.icm.train()


    def aug_and_encode(self, obs):
        obs = self.aug(obs)
        obs = self.encoder(obs)
        return obs
    
    def compute_intr_reward(self, obs, action, objects):
        rep = self.icm.get_rep(obs)
        reward = self.pbe(rep)
        reward = reward.reshape(-1, 1)
        return reward
    
    def train_icm(self, obs, action, obj_id, next_obs):
        metrics = dict()
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            self.icm_opt.zero_grad()
            forward_error, backward_error = self.icm(obs, action, obj_id, next_obs)
            avg_forward_error = forward_error.mean()
            avg_backward_error = backward_error.mean()
            loss = avg_forward_error + avg_backward_error
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.step(self.icm_opt)
            self.grad_scaler.update()
        metrics['unsupervised/extrinsic_reward'] = loss.item()
        metrics['unsupervised/forward_error'] = avg_forward_error.item()
        metrics['unsupervised/backward_error'] = avg_backward_error.item()
        return metrics



    def train_offline_from_batch(self, frames, lang, action, obj_id, lengths_frames, lengths_lang, interact_mask, rewards, terminals, eval=False):
        # all we have to do is modify the rewards being passed and then use that to run super().train_offline_from_batch
        # we need to get the intrinsic rewards from the ICM
        batch_size = frames.shape[0]
        seq_length = action.shape[1]
        obs_flattened = frames[:, :-1].view(-1, 512, 7, 7)
        next_obs_flattened = frames[:, 1:].view(-1, 512, 7, 7)
        action_flattened = action.view(-1)
        obj_id_flattened = obj_id.view(-1)
        obs = self.aug_and_encode(obs_flattened)
        with torch.no_grad():
            next_obs = self.aug_and_encode(next_obs_flattened)
            intr_reward = self.compute_intr_reward(obs, action_flattened, obj_id_flattened)
            intr_reward = intr_reward.view(batch_size, seq_length, 1)
        assert intr_reward.shape == rewards.shape
        rewards = intr_reward 
        # replace the language embedding with 0
        lang = torch.zeros_like(lang)[..., 0]
        lengths_lang = torch.ones_like(lengths_lang)
        metrics = super().train_offline_from_batch(frames, lang, action, obj_id, lengths_frames, lengths_lang, interact_mask, rewards, terminals, eval)
        # now just train ICM
        icm_metrics = self.train_icm(obs, action_flattened, obj_id_flattened, next_obs)
        metrics.update(icm_metrics)
        return metrics
    


class ICM(nn.Module):
    """
    Same as ICM, with a trunk to save memory for KNN
    """
    def __init__(self, obs_dim, num_objects, hidden_dim, icm_rep_dim):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(obs_dim, icm_rep_dim),
                                   nn.LayerNorm(icm_rep_dim), nn.Tanh())
        self.emb_action = nn.Embedding(13, hidden_dim//2)
        self.emb_object = nn.Embedding(num_objects + 1, hidden_dim//2)
        self.forward_net = nn.Sequential(
            nn.Linear(icm_rep_dim + hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, icm_rep_dim))

        self.backward_net = nn.Sequential(
            nn.Linear(2 * icm_rep_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh())

        self.apply(utils.weight_init)

    def forward(self, obs, action, objects, next_obs):
        assert obs.shape[0] == next_obs.shape[0]
        assert obs.shape[0] == action.shape[0]
        assert obs.shape[0] == objects.shape[0]
        obs = self.trunk(obs)
        next_obs = self.trunk(next_obs)
        action_embedding = self.emb_action(action)
        object_embedding = self.emb_object(objects)
        next_obs_hat = self.forward_net(torch.cat([obs, action_embedding, object_embedding], dim=-1))
        action_hat = self.backward_net(torch.cat([obs, next_obs], dim=-1))

        forward_error = torch.norm(next_obs - next_obs_hat,
                                   dim=-1,
                                   p=2,
                                   keepdim=True)
        backward_error = torch.norm(action - action_hat,
                                    dim=-1,
                                    p=2,
                                    keepdim=True)

        return forward_error, backward_error

    def get_rep(self, obs):
        rep = self.trunk(obs)
        return rep