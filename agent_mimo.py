from myimports import *
from models import *

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

class mimo_dataset(Dataset):
    def __init__(self, datadir, Ntrain=20, M=3, training=True):
        self.datadir = datadir
        self.Ntrain = Ntrain
        self.M = M
        
        if training:
            train_dict_fpath = self.datadir + 'traindata_dict_'+\
                             str(self.Ntrain)+'.pickle'
        
        else:
            train_dict_fpath = self.datadir + 'valdata_dict_500.pickle'
            self.Ntrain=500
            
            
        with open(train_dict_fpath,'rb') as file:
            traindata_dict = pickle.load(file)
            
        self.X_train = torch.from_numpy(traindata_dict['X'])
        self.Y_train = torch.from_numpy(traindata_dict['Y']) 
        self.Y_gt_train = torch.from_numpy(traindata_dict['Y_gt'])
        
    def __len__(self):
        return self.Ntrain
    
    def __getitem__(self, index):
        
        x_in = torch.tensor([self.X_train[index]])
        y_in = torch.tensor([self.Y_train[index]])
        
        for m in range(1,self.M):
            random_index = np.random.randint(0,self.Ntrain)
            xs = torch.tensor([self.X_train[random_index]])
            ys = torch.tensor([self.Y_train[random_index]])
            x_in = torch.cat((x_in, xs), axis=0)
            y_in = torch.cat((y_in, ys), axis=0)
        
        return x_in, y_in
    
#########################################################################
#########################################################################


class agent_mimo():
    def __init__(self, Ntrain, M=3, m_complexity=0, H=50,\
                 datadir = './datasets/'):
        
        device = torch.device('cuda:0' if torch.cuda.is_available() \
                              else 'cpu')
        self.device  = device
        self.datadir = datadir
        self.Ntrain = Ntrain
        self.M = M
        
        if m_complexity == 0:
            self.model = model_mimo(M=M, H=H).to(device)
        else:
            self.model = model_mimo2(M=M, H=H).to(device)
            
        self.learning_rate = 1e-2
        self.loss=nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),\
                                          lr = self.learning_rate)
        self.batch_size = 5
        self.num_epochs = 0
        
        self.training_loss_list   = []
        self.validation_loss_list = []
        
    

        
    def create_datasets(self):
        self.train_dset = mimo_dataset(self.datadir,\
                                       self.Ntrain,\
                                       self.M, training=True)
        self.val_dset   = mimo_dataset(self.datadir,\
                                       self.Ntrain,\
                                       self.M,training=False)
    
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
                y = y.view(-1,self.M).float()
                y_pred = self.model(x.view(-1,self.M).float())
                validation_loss+=self.loss(y_pred,y)
            validation_loss/=len(self.val_dloader)
            self.validation_loss_list.append(validation_loss)
        return validation_loss
    
    def train(self, num_epochs=40):
        for epoch in range(num_epochs):
            self.model.train()
            start_t = time.time()
            training_loss = 0
            
            for idx, (x,y) in enumerate(self.train_dloader):
                y = y.view(-1,self.M).float()
                y_pred = self.model(x.view(-1,self.M).float())
                
                if epoch==0 and idx == 0:
                    print('x.shape, y.shape: ', x.shape, y.shape)
                    print('y_pred.shape:', y_pred.shape)
                    
                loss = self.loss(y_pred,y)
                
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
        
    def test(self,test_xl=-7, test_xh=7, Ntest=100):
        self.Ntest = Ntest
        self.xl = test_xl
        self.xh = test_xh
        
        self.xs = np.linspace(self.xl, self.xh, self.Ntest, endpoint=True)
        self.ys = self.f(self.xs)
        
        self.pred_means = np.zeros((self.Ntest))
        self.pred_stds  = np.zeros((self.Ntest))
        self.pred_errs  = np.zeros((self.Ntest))
        
        self.pmeans_in, self.pmeans_out = [],[]
        self.pstds_in , self.pstds_out  = [],[]
        self.perrs_in , self.perrs_out  = [],[]
        self.xs_in,    self.xs_out = [],[]
        
        for idx in range(self.xs.shape[0]):
            pred_mean, pred_std  = self.predict(self.xs[idx])
            err = np.abs(self.ys[idx]-pred_mean)
            self.pred_means[idx] = pred_mean
            self.pred_stds[idx]  = pred_std
            self.pred_errs[idx]  = err
            
            #log ood and id seperately
            if self.xs[idx]>=-4 and self.xs[idx]<=4:
                self.xs_in.append(self.xs[idx])
                self.pmeans_in.append(pred_mean)
                self.pstds_in.append(pred_std)
                self.perrs_in.append(err)
            else:
                self.xs_out.append(self.xs[idx])
                self.pmeans_out.append(pred_mean)
                self.pstds_out.append(pred_std)
                self.perrs_out.append(err)
        
    
    def f(self, x):
        return x*x*x
    
    def predict(self,x_input):
        self.model.eval()
        x_in = torch.tensor([x_input for idx in range(self.M)]).float()
        y_pred = self.model(x_in)
        y_pred = y_pred.detach().cpu().squeeze().numpy()
        y_pred_mean = y_pred.mean()
        y_pred_std  = y_pred.std()
        
        return y_pred_mean, y_pred_std
    
    def plot_tlvl(self):
        fig = plt.figure(figsize=(8,4))
        plt.yscale('log')
        plt.plot(self.training_loss_list, label='training_loss')
        plt.plot(self.validation_loss_list, label='validation_loss')
        plt.xlabel('# Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        
    def plot_testing(self, show_datapoints=False):
        fig = plt.figure(figsize=(4,4))
        plt.plot(self.xs, self.ys, 'k', linewidth=1, alpha=1, label='f(x)')
        plt.vlines(-4, -255, 255, 'r', alpha=0.2)
        plt.vlines( 4, -255, 255, 'r', alpha=0.2)
        plt.fill_betweenx(np.arange(-255,255,0.1), -4, 4, \
                          alpha=0.05, color='r')
        
        plt.plot(self.xs, self.pred_means, '--C0',\
                                        linewidth=1,\
                                        alpha=1,label='pred')
        plt.fill_between(self.xs,\
                         self.pred_means - 2*self.pred_stds,\
                         self.pred_means + 2*self.pred_stds,\
                         label='pred+-2std',alpha=0.5)

        plt.ylim([-225,225])
        plt.xlim([self.xl+1, self.xh-1])
        plt.xlabel('input')
        plt.ylabel('prediction')
        
        if show_datapoints:
            plt.plot(self.X_train.numpy(), self.Y_train.numpy(), 'og', label='data points')
        
        plt.legend(loc='upper right')
        
    def plot_calibration(self, fit_line=True):
        fig = plt.figure(figsize=(4,4))
        plt.plot(self.pred_stds,self.pred_errs, 'o')
        plt.xlabel('prediction std')
        plt.ylabel('absolute prediction error')
        
        if fit_line:
            errs = self.pred_errs.reshape(-1,1).copy()
            stds = self.pred_stds.reshape(-1,1).copy()
            model = LinearRegression().fit(stds,errs )
            score = model.score(stds, errs)
            errs_pred = model.predict(stds)
            plt.plot(stds.squeeze(), errs_pred.squeeze(),
                     '--C1', label='R2_score:{:.3f}'.format(score))
        
            plt.legend(loc='upper right')
        plt.show()    
        
    def plot_idood_calibration(self,fit_line=False):
        pstds_in = np.array(self.pstds_in)
        perrs_in = np.array(self.perrs_in)
        
        pstds_out = np.array(self.pstds_out)
        perrs_out = np.array(self.perrs_out)
        
        fig = plt.figure(figsize=(4,4))
        plt.plot(pstds_in, perrs_in,\
        'ok',\
        label='id_std_mean={:.2f}'.format(pstds_in.mean()))
        
        plt.plot(pstds_out, perrs_out,\
        'oC3',\
        label='ood_std_mean={:.2f}'.format(pstds_out.mean()))
        
        plt.xlabel('prediction std')
        plt.ylabel('absolute prediction error')
        
        if fit_line:
            errs_in = perrs_in.reshape(-1,1).copy()
            stds_in = pstds_in.reshape(-1,1).copy()
            model_in = LinearRegression().fit(stds_in,errs_in )
            score_in = model_in.score(stds_in, errs_in)
            errs_in_pred = model_in.predict(stds_in)
            
            plt.plot(stds_in.squeeze(), errs_in_pred.squeeze(),
                     '--k', label='idR2_score:{:.3f}'.format(score_in))
            
            errs_out = perrs_out.reshape(-1,1).copy()
            stds_out = pstds_out.reshape(-1,1).copy()
            model_out = LinearRegression().fit(stds_out,errs_out)
            score_out = model_out.score(stds_out, errs_out)
            errs_out_pred = model_out.predict(stds_out)
            
            plt.plot(stds_out.squeeze(), errs_out_pred.squeeze(),
                     '--C3', label='oodR2_score:{:.3f}'.format(score_out))
        
        plt.legend(loc='upper left')
        plt.show()
            
    