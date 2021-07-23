from myimports import *

class model_de(nn.Module):
    def __init__(self, D_in = 1, D_out=1, H=50):
        super(model_de, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
        )
        self.encode_mu = nn.Sequential(
            nn.Linear(H,D_out)
        )
        self.encode_logvar = nn.Sequential(
            nn.Linear(H,D_out)
        )
        
    def forward(self, x):
        code = self.fc(x)
        mu = self.encode_mu(code)
        logvar = self.encode_logvar(code)
        
        return mu, logvar
    
class model_de2(nn.Module):
    def __init__(self, D_in = 1, D_out=1, H=50):
        super(model_de2, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Linear(H, H),
            nn.ReLU(),
        )
        self.encode_mu = nn.Sequential(
            nn.Linear(H,D_out)
        )
        self.encode_logvar = nn.Sequential(
            nn.Linear(H,D_out)
        )
        
    def forward(self, x):
        code = self.fc(x)
        mu = self.encode_mu(code)
        logvar = self.encode_logvar(code)
        
        return mu, logvar
    
#################################################################
#################################################################

class model_mcdrop(nn.Module):
    def __init__(self, D_in = 1, D_out=1, H=50,dropout_rate=0.2):
        super(model_mcdrop, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(H,D_out)
        )
        
    def forward(self, x):
        return self.model(x)
    
class model_mcdrop2(nn.Module):
    def __init__(self, D_in = 1, D_out=1, H=50,dropout_rate=0.2):
        super(model_mcdrop2, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            
            nn.Linear(H, H),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            
            nn.Linear(H,D_out)
        )
        
    def forward(self, x):
        return self.model(x)
    
    
#################################################################
#################################################################   

class model_mimo(nn.Module):
    def __init__(self, M=3, H=50):
        super(model_mimo, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(M, H),
            nn.ReLU(),
            nn.Linear(H,M)
        )
        
    def forward(self, x):
        return self.model(x)
    
class model_mimo2(nn.Module):
    def __init__(self, M=3, H=50):
        super(model_mimo2, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(M, H),
            nn.ReLU(),
            
            nn.Linear(H, H),
            nn.ReLU(),
            
            nn.Linear(H,M)
        )
        
    def forward(self, x):
        return self.model(x)
