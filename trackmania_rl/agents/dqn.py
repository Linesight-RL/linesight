import torch


class Agent(torch.nn.Module):
    def __init__(self, float_inputs_dim, float_hidden_dim):
        super().__init__()
        linear = torch.nn.Linear
        self.img_head = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(16, 16), stride=8),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(8, 8), stride=4),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=2),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Flatten(),
        )
        self.float_feature_extractor = torch.nn.Sequential(
            torch.nn.Linear(float_inputs_dim, float_hidden_dim),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Linear(float_hidden_dim, float_hidden_dim),
            torch.nn.LeakyReLU(inplace=True),
        )
        CNN_Out_Dimension = 1152
        Dense_Input_Dimension = CNN_Out_Dimension + Float_Feature_Extractor_Width
        self.lrelu = torch.nn.LeakyReLU()
        if Architecture == "DQN":
            self.dense_head = torch.nn.Sequential(
                linear(Dense_Input_Dimension, Linear_Width),
                torch.nn.LeakyReLU(inplace=True),
                linear(Linear_Width, len(Actions)),
            )
        else:  # Duelnet
            Linear_Half_Width = Linear_Width // 2
            self.A_head = torch.nn.Sequential(
                linear(Dense_Input_Dimension, Linear_Half_Width),
                torch.nn.LeakyReLU(inplace=True),
                linear(Linear_Half_Width, len(Actions)),
            )
            self.V_head = torch.nn.Sequential(
                linear(Dense_Input_Dimension, Linear_Half_Width),
                torch.nn.LeakyReLU(inplace=True),
                linear(Linear_Half_Width, 1),
            )
        if (
            Learning_Mode == "IQN"
        ):  # Inspiration from https://github.com/valeoai/rainbow-iqn-apex/blob/master/rainbowiqn/model.py
            self.iqn_fc = torch.nn.Linear(
                IQN_Embedding_Dimension, Dense_Input_Dimension
            )  # There is no word in the paper on how to init this layer?
        self.Initialise_Weights()

    def Initialise_Weights(self):
        for m in self.img_head:
            if isinstance(m, torch.nn.Conv2d):
                Init_Kaiming(m)
        for m in self.float_feature_extractor:
            if isinstance(m, torch.nn.Linear):
                Init_Kaiming(m)
        if Exploration_Mode != "NoisyNet":
            if Architecture == "DQN":
                Init_Kaiming(self.dense_head[0])
                Init_Xavier(self.dense_head[2])
            else:  # Duelnet
                Init_Kaiming(self.A_head[0])
                Init_Kaiming(self.V_head[0])
                Init_Xavier(self.A_head[2])
                Init_Xavier(self.V_head[2])

    def forward(self, img, float_inputs, num_quantiles, return_Q, tau=None):
        img = (img.float() - 128) / 128
        img_outputs = self.img_head(img)
        float_outputs = self.float_feature_extractor(float_inputs)
        concat = torch.cat((img_outputs, float_outputs), 1)
        if Learning_Mode == "IQN":
            if tau is None:
                tau = torch.cuda.FloatTensor(img.shape[0] * num_quantiles, 1).uniform_(0, 1)
            quantile_net = tau.expand([-1, IQN_Embedding_Dimension])
            quantile_net = torch.cos(
                torch.arange(1, IQN_Embedding_Dimension + 1, 1, device="cuda", dtype=torch.float32)
                * math.pi
                * quantile_net
            )
            quantile_net = self.iqn_fc(quantile_net)
            quantile_net = self.lrelu(quantile_net)
            concat = concat.repeat(num_quantiles, 1)
            concat = concat * quantile_net
        if Architecture == "DQN":
            Q = self.dense_head(concat)
        else:  # Implementation inspired from https://pytorch.org/rl/_modules/torchrl/modules/models/models.html#DuelingCnnDQNet and https://github.com/ray-project/ray/blob/master/rllib/algorithms/dqn/dqn_torch_model.py
            A = self.A_head(concat)
            if return_Q:
                V = self.V_head(concat)
                Q = V + A - A.mean(dim=-1, keepdim=True)
            else:
                Q = A
        if Learning_Mode == "DQN":
            return Q
        else:  # IQN
            return Q, tau

    def reset_noise(self):
        if Architecture == "DQN":
            self.dense_head[0].reset_noise()
            self.dense_head[2].reset_noise()
        else:
            self.A_head[0].reset_noise()
            self.A_head[2].reset_noise()
            self.V_head[0].reset_noise()
            self.V_head[2].reset_noise()
