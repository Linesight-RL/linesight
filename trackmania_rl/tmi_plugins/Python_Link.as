Net::Socket@ sock = null;
Net::Socket@ clientSock = null;

enum MessageType {
    SCRunStepSync = 1,
    SCCheckpointCountChangedSync = 2,
    SCLapCountChangedSync = 3,
    SCRequestedFrameSync = 4,
    CSetSpeed = 5,
    CRewindToState = 6,
    CGetSimulationState = 7,
    CSetInputState = 8,
    CGiveUp = 9,
    CPreventSimulationFinish = 10,
    CShutdown = 11,
    CExecuteCommand = 12,
    CSetTimeout = 13,
    CRaceFinished = 14,
    CRequestFrame = 15,
}

const bool debug = false;
const string HOST = "127.0.0.1";
const uint16 PORT = 8477;
uint RESPONSE_TIMEOUT = 2000;
int next_frame_requested = -1;

void WaitForResponse(MessageType type){
    auto now = Time::Now;

    while (true) {
        auto receivedType = HandleMessage();
        if (receivedType == type) {
            break;
        }

        if (receivedType == MessageType::CShutdown) {
            break;
        }

        if (receivedType == -1 && Time::Now - now > RESPONSE_TIMEOUT) {
            log("Client disconnected due to timeout (" + RESPONSE_TIMEOUT + "ms)");
            @clientSock = null;
            break;
        }
    }
}

int HandleMessage()
{
    if (clientSock.Available == 0) {
        return -1;
    }

    int type = clientSock.ReadInt32();
    switch(type) {
        case MessageType::SCRunStepSync: {
            break;
        }

        case MessageType::SCRequestedFrameSync: {
            break;
        }

        case MessageType::SCCheckpointCountChangedSync: {
            break;
        }

        case MessageType::CSetSpeed: {
            auto@ simManager = GetSimulationManager();
            if(debug){
                print("Server: SetSpeed message");
            }
            float new_speed = clientSock.ReadFloat();
            simManager.SetSpeed(new_speed);
            if(debug){
                print("Server: Set speed to "+new_speed);
            }
            break;
        }

        case MessageType::CGiveUp: {
            auto@ simManager = GetSimulationManager();
            if(debug){
                print("Server: Give up");
            }
            if (simManager.InRace) {
                simManager.GiveUp();
            }
            break;
        }

        case MessageType::CPreventSimulationFinish: {
            auto@ simManager = GetSimulationManager();
            if(debug){
                print("Server: prevent simulation finish");
            }
            if (simManager.InRace) {
                simManager.PreventSimulationFinish();
            }
            break;
        }

        case MessageType::CRewindToState: {
            int32 stateLength = clientSock.ReadInt32();
            auto stateData = clientSock.ReadBytes(stateLength);
            auto@ simManager = GetSimulationManager();
            if(debug){
                print("Server: rewind message");
            }
            if (simManager.InRace) {
                SimulationState state(stateData);
                simManager.RewindToState(state);
            }
            break;            
        }


        case MessageType::CGetSimulationState: {
            auto@ simManager = GetSimulationManager();
            auto@ state = simManager.SaveState();
            auto@ data = state.ToArray();
            if(debug){
                print("Server: get_simulation_state");
            }

            clientSock.Write(int(data.Length));
            clientSock.Write(data);
            break;
        }

        case MessageType::CSetInputState: {
            if(debug){
                print("Server: Set input state message");
            }
            bool left = clientSock.ReadUint8()>0;
            bool right = clientSock.ReadUint8()>0;
            bool accelerate = clientSock.ReadUint8()>0;
            bool brake = clientSock.ReadUint8()>0;

            if(debug){
                print("Set input state to "+left+right+accelerate+brake);
            }

            auto@ simManager = GetSimulationManager();
            if (simManager.InRace) {
                simManager.SetInputState(InputType::Left, left?1:0);
                simManager.SetInputState(InputType::Right, right?1:0);
                simManager.SetInputState(InputType::Up, accelerate?1:0);
                simManager.SetInputState(InputType::Down, brake?1:0);
            }

            break;
        }

        case MessageType::CShutdown: {
            log("Client disconnected");
            @clientSock = null;
            break;
        }

        case MessageType::CExecuteCommand: {
            const int32 bytes_to_read = clientSock.ReadInt32();
            const string command = clientSock.ReadString(bytes_to_read);
            if(debug){
                print("Server: command "+command+" received");
            }
            ExecuteCommand(command);
            break;
        }

        case MessageType::CSetTimeout: {
            const uint new_timeout = clientSock.ReadUint32();
            if(debug){
                print("Server: set timeout to "+new_timeout);
            }
            RESPONSE_TIMEOUT = new_timeout;
            break;
        }

        case MessageType::CRaceFinished: {
            auto@ simManager = GetSimulationManager();
            int is_race_finished = (simManager.PlayerInfo.RaceFinished?1:0);
            if(debug){
                print("Server: Answering race_finished with "+is_race_finished);
            }
            clientSock.Write(is_race_finished);
            break;
        }

        case MessageType::CRequestFrame: {
            next_frame_requested = clientSock.ReadInt32();
            if(debug){
                print("Client requested frame after "+next_frame_requested+" skip");
            }
            break;
        }

        default: {
            print("Server: got unknown message "+type);
            break;
        }
    }

    return type;
}

void OnRunStep(SimulationManager@ simManager){
    if (@clientSock is null) {
        return;
    }
    if(debug){
        print("Server: OnRunStep");
    }
    auto@ state = simManager.SaveState();

    clientSock.Write(MessageType::SCRunStepSync);
    clientSock.Write(state.get_PlayerInfo().RaceTime);
    WaitForResponse(MessageType::SCRunStepSync);
}

void OnCheckpointCountChanged(SimulationManager@ simManager, int current, int target){
    if (@clientSock is null) {
        return;
    }
    if(debug){
        print("Server: OnCheckpointCountChanged");
    }

    clientSock.Write(MessageType::SCCheckpointCountChangedSync);
    clientSock.Write(current);
    clientSock.Write(target);
    WaitForResponse(MessageType::SCCheckpointCountChangedSync);
}

void OnLapCountChanged(SimulationManager@ simManager, int current, int target){
    if (@clientSock is null) {
        return;
    }
    if(debug){
        print("Server: OnLapCountChanged");
    }

    clientSock.Write(MessageType::SCLapCountChangedSync);
    clientSock.Write(current);
    clientSock.Write(target);
    WaitForResponse(MessageType::SCLapCountChangedSync);
}

void Main(){
    if (@sock is null) {
        @sock = Net::Socket();
        sock.Listen(HOST, PORT);
    }
}

void Render(){
    //Bluescreens if you print every Render()
    auto @newSock = sock.Accept(0);
    if (@newSock !is null) {
        @clientSock = @newSock;
        log("Client connected (IP: " + clientSock.RemoteIP + ")");
    }
    if(next_frame_requested==0){
        next_frame_requested = -1;
        auto@ simManager = GetSimulationManager();
        auto@ state = simManager.SaveState();
        if(debug){
            print("Notifying of frame at race_time "+state.get_PlayerInfo().RaceTime);
        }
        clientSock.Write(MessageType::SCRequestedFrameSync);
        WaitForResponse(MessageType::SCRequestedFrameSync);
        if(debug){
            print("Got response from client for SCRequestedFrameSync");
        }
    }
    else if(next_frame_requested>-1){
        --next_frame_requested;
    }
}

PluginInfo@ GetPluginInfo(){
    PluginInfo info;
    info.Author = "Agade";
    info.Name = "Python Link";
    info.Description = "Reproduce close to original TMI <2 python interface with TMI 2.1 sockets";
    info.Version = "0.1";
    return info;
}