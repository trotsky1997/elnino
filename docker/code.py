import main,torch,os,shutil
import numpy as np
model = main.p2e()

if torch.cuda.is_available():
    model.load_state_dict(torch.load("./model.pt"))
    model = model.cuda()
else:
    model.load_state_dict(torch.load("./model.pt",map_location=torch.device('cpu')))
model.eval()
with torch.no_grad():
    for root, dirs, files in os.walk("./tcdata/", topdown=False):
        for name in files:
            f = os.path.join(root, name)
            test_data = torch.tensor(np.load(f)).permute(3,0,1,2).unsqueeze(0).float()
            if torch.cuda.is_available():
                test_data = test_data.cuda()
            res = model(test_data).squeeze(0).detach().cpu().numpy()
            np.save('./result/' + name, res)
            print(res)
