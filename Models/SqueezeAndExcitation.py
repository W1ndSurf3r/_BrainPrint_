from torch import nn


class SELayer(nn.Module):
    def __init__(self, channel, input_size = 400 , reduction=2):
        super(SELayer, self).__init__()
        #self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.avg_pool = nn.AvgPool2d((1,400), stride=(1,400))
        
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, _, c, _ = x.size()
        #print('step 1',x.size())
        y = self.avg_pool(x)
        #print('step 2',y.size())
        y = y.view(b, c)
        #print('step 3',y.size())
        y = self.fc(y)
        #print('step 4',y.size())
        y = y.view(b, 1, c, 1)
        #print('step 5',y.size(), x.size())
        #print(y[0,0], y[0,1])
        #print(x.size())
        #y = y.expand_as(x)
        #print(x[0,1])
        #print(x[2,1])
        return x * y.expand_as(x), y