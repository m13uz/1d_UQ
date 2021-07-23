from myimports import *
from models import *

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

class agent_de():
    def __init__(self, Ntrain, datadir = './datasets/', \
                 m_complexity=0, H=50 ):
        
        device = torch.device('cuda:0' if torch.cuda.is_available() \
                              else 'cpu')
        self.device = device
        self.datadir = datadir
        
        self.Ntrain=Ntrain
        
        if m_complexity == 0:
            self.model = model_de(H=H).to(device)
        else:
            self.model = model_de2(H=H).to(device)

        
        self.learning_rate = 1e-2
        self.optimizer     = torch.optim.Adam(self.model.parameters(), \
                                              lr=self.learning_rate)
        self.batch_size = 5
        self.num_epochs = 0
        
        self.training_loss_list   = []
        self.validation_loss_list = []
        
    def nll_loss(self, label, mu, logvar):
        NLL = torch.sum(logvar/2) + \
            torch.sum((label-mu)*(label-mu)/2/torch.exp(logvar))
        return NLL
    
    def create_datasets(self):
        #create the training dataset object
        train_dict_fpath = self.datadir + 'traindata_dict_'+\
                         str(self.Ntrain)+'.pickle'
        
        with open(train_dict_fpath,'rb') as file:
            traindata_dict = pickle.load(file)
            
        self.X_train = torch.from_numpy(traindata_dict['X'])
        self.Y_train = torch.from_numpy(traindata_dict['Y']) 
        self.Y_gt_train = torch.from_numpy(traindata_dict['Y_gt'])
        
        self.train_dset = TensorDataset(self.X_train, self.Y_train)
        
        #create the validation dataset object
        val_dict_fpath = self.datadir + 'valdata_dict_500.pickle'
        
        with open(val_dict_fpath,'rb') as file:
            valdata_dict = pickle.load(file)
            
        self.X_val = torch.from_numpy(valdata_dict['X'])
        self.Y_val = torch.from_numpy(valdata_dict['Y']) 
        self.Y_gt_val = torch.from_numpy(valdata_dict['Y_gt'])
        
        self.val_dset = TensorDataset(self.X_val, self.Y_val)
        
        

    def create_dataloaders(self):
        self.train_dloader = DataLoader(self.train_dset, \
                                       batch_size=self.batch_size,\
                                       shuffle=True)
        self.val_dloader   = DataLoader(self.val_dset, \
                                       batch_size=self.batch_size,\
                                       shuffle=False) 
    
    def validate(self):
        with torch.no_grad():
            self.model.eval()
            validation_loss=0
            for idx, (x,y) in enumerate(self.val_dloader):
                y = y.view(-1,1).float()
                mu, logvar = self.model(x.view(-1,1).float())
                validation_loss+=self.nll_loss(y, mu, logvar)
            validation_loss/=len(self.val_dloader)
            self.validation_loss_list.append(validation_loss)
        return validation_loss

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            self.model.train()
            start_t = time.time()
            training_loss = 0
            
            for idx, (x,y) in enumerate(self.train_dloader):
                y = y.view(-1,1).float()
                mu, logvar = self.model(x.view(-1,1).float())
                
                if epoch==0 and idx == 0:
                    print('x.shape, y.shape: ', x.shape, y.shape)
                    print('mu.shape, logvar.shape', mu.shape, logvar.shape)
                    
                loss = self.nll_loss(y, mu, logvar)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                training_loss+=loss.item()
                
            training_loss/=len(self.train_dloader)
            self.training_loss_list.append(training_loss)
            end_t = time.time()
            
            #validate the current model
            validation_loss = self.validate()
            
            #log
            if (epoch+1)%10 == 0:
                 print('Epoch [{}/{}], t_loss: {:.7f}, val_loss: {:.7f},\
                  time: {:.2f}'.format(
                        epoch+1,self.num_epochs+epoch,\
                        training_loss,\
                        validation_loss,\
                        end_t-start_t))
            
        self.num_epochs +=num_epochs
        
        self.plot_tlvl()
         
    def predict(self, x_input):
        x_input_tensor = torch.tensor(x_input).view(-1,1).float()
        self.model.eval()
        mu, logvar = self.model(x_input_tensor)
        mu_npa = mu.detach().cpu().squeeze().numpy()
        logvar_npa = logvar.detach().cpu().squeeze().numpy()
        
        return mu_npa, logvar_npa
    
    
    def f(self, x):
        return x*x*x
    
    def test(self, test_xl=-7, test_xh=7, Ntest=100):
        self.Ntest = Ntest
        self.xl = test_xl
        self.xh = test_xh
        
        self.xs = np.linspace(self.xl, self.xh, self.Ntest, endpoint=True)
        self.ys = self.f(self.xs)
        
        self.mus = np.zeros((self.Ntest))
        self.logvars = np.zeros((self.Ntest))
        self.errs = np.zeros((self.Ntest))
        
        self.mus_in, self.mus_out = [],[]
        self.logvars_in, self.logvars_out = [],[]
        self.errs_in, self.errs_out = [],[]
        self.xs_in, self.xs_out = [],[]
        
        for idx in range(self.xs.shape[0]):
            mu, logvar   = self.predict(self.xs[idx])
            err = np.abs(self.ys[idx]-mu)
            self.mus[idx]     = mu
            self.logvars[idx] = logvar
            self.errs[idx]    = err
            
            if self.xs[idx]>=-4 and self.xs[idx]<=4:
                self.mus_in.append(mu)
                self.logvars_in.append(logvar)
                self.errs_in.append(err)
                self.xs_in.append(self.xs[idx])
            else:
                self.mus_out.append(mu)
                self.logvars_out.append(logvar)
                self.errs_out.append(err)
                self.xs_out.append(self.xs[idx])
        
        self.mus_in  = np.array(self.mus_in)
        self.mus_out = np.array(self.mus_out)
        self.logvars_in = np.array(self.logvars_in)
        self.logvars_out= np.array(self.logvars_out)
        self.errs_in  = np.array(self.errs_in)
        self.errs_out = np.array(self.errs_out)
        
        self.stds    = np.exp(0.5*self.logvars)
        self.stds_in = np.exp(0.5*self.logvars_in)
        self.stds_out= np.exp(0.5*self.logvars_out)
        
        return self.mus, self.logvars, self.stds, self.errs

    
    
    def plot_tlvl(self):
        fig = plt.figure(figsize=(8,4))
        plt.yscale('log')
        plt.plot(self.training_loss_list, label='training_loss')
        plt.plot(self.validation_loss_list, label='validation_loss')
        plt.xlabel('# Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        
    def plot_testing(self, show_datapoints = False):
        fig = plt.figure(figsize=(4,4))
        plt.plot(self.xs, self.ys, 'k', linewidth=1, alpha=1, label='f(x)')
        plt.vlines(-4, -255, 255, 'r', alpha=0.2)
        plt.vlines( 4, -255, 255, 'r', alpha=0.2)
        plt.fill_betweenx(np.arange(-255,255,0.1), -4, 4, \
                          alpha=0.05, color='r')
        
        plt.plot(self.xs, self.mus, '--C0',linewidth=1, alpha=1, label='pred')
        plt.fill_between(self.xs,\
                         self.mus - 2*np.exp(0.5*self.logvars),\
                         self.mus + 2*np.exp(0.5*self.logvars),\
                         label='pred+-2std',alpha=0.5)

        plt.ylim([-225,225])
        plt.xlim([self.xl+1, self.xh-1])
        plt.xlabel('input')
        plt.ylabel('prediction')
        
        if show_datapoints:
            plt.plot(self.X_train.numpy(), self.Y_train.numpy(), 'og', label='data points')
        
        plt.legend(loc='upper right')
        
    def plot_calibration(self, fit_line=False):
        fig = plt.figure(figsize=(4,4))
        plt.plot(self.stds,self.errs, 'o')
        plt.xlabel('prediction std')
        plt.ylabel('absolute prediction error')
        
        if fit_line:
            errs = self.errs.reshape(-1,1).copy()
            stds = self.stds.reshape(-1,1).copy()
            print('errs.shape: ', errs.shape)
            print('stds.shape: ', stds.shape)
            model = LinearRegression().fit(stds,errs )
            score = model.score(stds, errs)
            errs_pred = model.predict(stds)
            plt.plot(stds.squeeze(), errs_pred.squeeze(),
                     '--C1', label='R2_score:{:.3f}'.format(score))
        
            plt.legend()
        plt.show()
        
    def plot_idood_calibration(self, fit_line=False):
        fig = plt.figure(figsize=(4,4))
        plt.plot(self.stds_in,self.errs_in, 'ko',\
        label = 'id_std_mean={:.3f}'.format(self.stds_in.mean()))
        plt.plot(self.stds_out,self.errs_out, 'oC3',\
        label = 'ood_std_mean={:.3f}'.format(self.stds_out.mean())) 
        plt.xlabel('prediction std')
        plt.ylabel('absolute prediction error')
        
        if fit_line:
            stds_in = self.stds_in.reshape(-1,1).copy()
            errs_in = self.errs_in.reshape(-1,1).copy()
            model_in = LinearRegression().fit(stds_in,errs_in)
            score_in = model_in.score(stds_in, errs_in)
            
            errs_in_pred = model_in.predict(stds_in)
            plt.plot(stds_in.squeeze(), errs_in_pred.squeeze(),
                     '--k', label='idR2_score:{:.3f}'.format(score_in))
            
            stds_out = self.stds_out.reshape(-1,1).copy()
            errs_out = self.errs_out.reshape(-1,1).copy()
            model_out = LinearRegression().fit(stds_out,errs_out)
            score_out = model_out.score(stds_out, errs_out)
            
            errs_out_pred = model_out.predict(stds_out)
            plt.plot(stds_out.squeeze(), errs_out_pred.squeeze(),
                     '--C3', label='oodR2_score:{:.3f}'.format(score_out))
            
        
        plt.legend()
        plt.show()

        
class ensemble_de():
    def __init__(self, N_agents=3, Ntrain=20, m_complexity=0, H=50):
        self.N_agents = N_agents
        self.Ntrain = Ntrain
        self.agents = []
        
        for n in range(self.N_agents):
            de = agent_de(Ntrain=self.Ntrain,\
                          m_complexity=m_complexity,\
                          H=H)
            self.agents.append(de)
        
    def train(self,num_epochs=40):
        for agent in self.agents:
            agent.create_datasets()
            agent.create_dataloaders()
            agent.train(num_epochs)
    
        
    def f(self, x):
        return x*x*x
    
    def test_agents(self,test_xl=-7, test_xh=7, Ntest=100):
        for agent in self.agents:
            _,_,_,_ = agent.test(test_xl, test_xh, Ntest)
    
    def test_ensemble(self,test_xl=-7, test_xh=7, Ntest=100):
        self.Ntest = Ntest
        self.xl = test_xl
        self.xh = test_xh
        
        self.xs = np.linspace(self.xl, self.xh, self.Ntest, endpoint=True)
        self.ys = self.f(self.xs)
        
        self.mus = np.zeros((self.Ntest))
        self.stds = np.zeros((self.Ntest))
        self.errs = np.zeros((self.Ntest))
        
        self.mus_in, self.mus_out = [],[]
        self.stds_in, self.stds_out = [],[]
        self.errs_in, self.errs_out = [],[]
        self.xs_in, self.xs_out = [],[]
        
        for idx in range(self.xs.shape[0]):
            mu, std   = self.predict(self.xs[idx])
            err = np.abs(self.ys[idx]-mu)
            self.mus[idx]     = mu
            self.stds[idx]    = std
            self.errs[idx]    = err
            
            if self.xs[idx]>=-4 and self.xs[idx]<=4:
                self.mus_in.append(mu)
                self.stds_in.append(std)
                self.errs_in.append(err)
                self.xs_in.append(self.xs[idx])
            else:
                self.mus_out.append(mu)
                self.stds_out.append(std)
                self.errs_out.append(err)
                self.xs_out.append(self.xs[idx])
        
        self.mus_in  = np.array(self.mus_in)
        self.mus_out = np.array(self.mus_out)
        
        self.stds_in = np.array(self.stds_in)
        self.stds_out= np.array(self.stds_out)
        
        self.errs_in  = np.array(self.errs_in)
        self.errs_out = np.array(self.errs_out)
        
        
        return self.mus, self.stds, self.errs
    
    def predict(self, x_input):
        mus = np.zeros((self.N_agents))
        logvars = np.zeros((self.N_agents))
        stds = np.zeros((self.N_agents))
        
        for idx, agent in enumerate(self.agents):
            mu, logvar = agent.predict(x_input)
            std = np.exp(0.5*logvar)
            
            mus[idx] = mu
            logvars[idx] = logvar
            stds[idx] = std
            
        ens_mean = mus.mean()
        ens_var  = (stds*stds+mus*mus).mean()-ens_mean*ens_mean
        ens_std  = np.sqrt(ens_var) 
        
        return ens_mean, ens_std
    
    
    def plot_ensemble_testing(self, show_datapoints = False):
        fig = plt.figure(figsize=(4,4))
        plt.plot(self.xs, self.ys, 'k', linewidth=1, alpha=1, label='f(x)')
        plt.vlines(-4, -255, 255, 'r', alpha=0.2)
        plt.vlines( 4, -255, 255, 'r', alpha=0.2)
        plt.fill_betweenx(np.arange(-255,255,0.1), -4, 4, \
                          alpha=0.05, color='r')
        
        plt.plot(self.xs, self.mus, '--C0',linewidth=1, alpha=1, label='ens_pred')
        plt.fill_between(self.xs,\
                         self.mus - 2*self.stds,\
                         self.mus + 2*self.stds,\
                         label='ens_pred+-2std',alpha=0.5)

        plt.ylim([-225,225])
        plt.xlim([self.xl+1, self.xh-1])
        plt.xlabel('input')
        plt.ylabel('prediction')
        
        if show_datapoints:
            plt.plot(self.agents[0].X_train.numpy(),\
                     self.agents[0].Y_train.numpy(),\
                     'og', label='data points')
        
        plt.legend(loc='upper right')
            
    def plot_ensemble_calibration(self, fit_line=False):
        fig = plt.figure(figsize=(4,4))
        plt.plot(self.stds,self.errs, 'o')
        plt.xlabel('prediction std')
        plt.ylabel('absolute prediction error')
        
        if fit_line:
            errs = self.errs.reshape(-1,1).copy()
            stds = self.stds.reshape(-1,1).copy()
            print('errs.shape: ', errs.shape)
            print('stds.shape: ', stds.shape)
            model = LinearRegression().fit(stds,errs )
            score = model.score(stds, errs)
            errs_pred = model.predict(stds)
            plt.plot(stds.squeeze(), errs_pred.squeeze(),
                     '--C1', label='R2_score:{:.3f}'.format(score))
        
            plt.legend()
        plt.show()
        
    def plot_ensemble_idood_calibration(self, fit_line=False):
        fig = plt.figure(figsize=(4,4))
        plt.plot(self.stds_in,self.errs_in, 'ko',\
        label = 'id_std_mean={:.3f}'.format(self.stds_in.mean()))
        plt.plot(self.stds_out,self.errs_out, 'oC3',\
        label = 'ood_std_mean={:.3f}'.format(self.stds_out.mean())) 
        plt.xlabel('prediction std')
        plt.ylabel('absolute prediction error')
        
        if fit_line:
            stds_in = self.stds_in.reshape(-1,1).copy()
            errs_in = self.errs_in.reshape(-1,1).copy()
            model_in = LinearRegression().fit(stds_in,errs_in)
            score_in = model_in.score(stds_in, errs_in)
            
            errs_in_pred = model_in.predict(stds_in)
            plt.plot(stds_in.squeeze(), errs_in_pred.squeeze(),
                     '--k', label='idR2_score:{:.3f}'.format(score_in))
            
            stds_out = self.stds_out.reshape(-1,1).copy()
            errs_out = self.errs_out.reshape(-1,1).copy()
            model_out = LinearRegression().fit(stds_out,errs_out)
            score_out = model_out.score(stds_out, errs_out)
            
            errs_out_pred = model_out.predict(stds_out)
            plt.plot(stds_out.squeeze(), errs_out_pred.squeeze(),
                     '--C3', label='oodR2_score:{:.3f}'.format(score_out))
            
        
        plt.legend()
        plt.show()