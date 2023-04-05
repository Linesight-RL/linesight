Swap_Mode = ['pb4Swap','equality'][1]
Exploration_Mode = ['EpsilonGreedy','NoisyNet'][1]
Epsilon = 0.01
Architecture = DuelNet
Learning_Mode = IQN
AL_Mode = None
Use_DDQN = True
AL_alpha = 0.95

with torch.no_grad():
		with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            Rewards = Rewards[:, None].repeat([IQN_N, 1])
            Terminal = Terminal[:, None].repeat([IQN_N, 1])
            Action_Indexes = Action_Indexes[:, None].repeat([IQN_N, 1])
            Adjust_Noise(model,True)
            outputs_next_action, _ = model(Image_Inputs,Float_Inputs,IQN_N,False)
            outputs_next_action = outputs_next_action.reshape([IQN_N, Batch_Size, len(Actions)])
            outputs_next_action = torch.mean(outputs_next_action, dim=0)
            outputs_next_action = torch.argmax(outputs_next_action,dim=1)

            outputs_next_action = outputs_next_action[:, None].repeat([IQN_N, 1])
			Adjust_Noise(model2,True)
            outputs_target, tau_target = model2(Image_Inputs,Float_Inputs,IQN_N,True)
            outputs_target = torch.gather(outputs_target, 1, outputs_next_action)
			outputs_target = Rewards+pow(Gamma,N_Steps)*outputs_target
            outputs_target = torch.where(Terminal,Rewards,outputs_target)
            outputs_target = outputs_target.reshape([IQN_N, Batch_Size, 1])
            outputs_target = outputs_target.permute([1, 0, 2])
	with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
		Adjust_Noise(model,True)
        outputs, tau = model(Prev_Image_Inputs,Prev_Float_Inputs,IQN_N,True)
        outputs = torch.gather(outputs, 1, Action_Indexes.type(torch.int64))
        outputs = outputs.reshape([IQN_N, Batch_Size, 1])
        outputs = outputs.permute([1, 0, 2])
        TD_Error = outputs_target[:, :, None, :]-outputs[:, None, :, :]
        huber_loss_case_one = (torch.abs(TD_Error) <= IQN_Kappa).float() * 0.5 * TD_Error ** 2
        huber_loss_case_two = (torch.abs(TD_Error) > IQN_Kappa).float() * IQN_Kappa * (torch.abs(TD_Error) - 0.5 * IQN_Kappa)
        loss = huber_loss_case_one + huber_loss_case_two
        tau = torch.reshape(tau, [IQN_N, Batch_Size, 1])
        tau = tau.permute([1, 0, 2])
        tau = tau[:, None, :, :].expand([-1, IQN_N, -1, -1])
        loss = (torch.abs(tau - ((TD_Error < 0).float()).detach()) * loss) / IQN_Kappa
        loss = torch.sum(loss, dim=2)
        loss = torch.mean(loss,dim=1)
        loss = loss[:,0]
		total_loss = torch.sum(Training_Batch_IS_Weights*loss)
	optimizer.zero_grad(set_to_none=True)
	scaler.scale(total_loss).backward()
	scaler.step(optimizer)
	scaler.update()
	return total_loss.detach().cpu(), loss.detach().cpu()