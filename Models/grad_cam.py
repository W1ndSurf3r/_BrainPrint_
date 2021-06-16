"""
Created on Thu Oct 26 11:06:51 2017
@author: Utku Ozbulak - github.com/utkuozbulak
"""
from PIL import Image
import numpy as np
import torch

from torch.autograd import Variable


class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer):
        
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.conv_outputs = None
    
    def save_gradient(self, grad):
        print('hooking up',grad.shape)
        self.gradients = grad
    
    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, x):
        return self.features_conv(x)
        
    def register_hook(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        self.model.eval()
        for index, module in enumerate(self.model.children()):
            #print('Selected: index {}, module {}'.format(index, module))
            if int(index) == self.target_layer:
                print('Selected: index {}, module {}'.format(index, module))
                x = x.permute(0, 1,4, 2, 3)
                x =  x.reshape(x.size(0)*11,x.size(2),x.size(3),x.size(4)) 
                x = module(x)  # Forward   
                h = x.register_hook(self.save_gradient) 
                self.conv_outputs = x
            elif index == 1:
                #print('Following: index {}, module {}'.format(index, module))
                x ,_ = module(x)
            elif index >=2 and index <=6 :
                #print('bad:', module)
                x= module(x)
            elif index ==7:
                #print(x.shape)
                x = x.reshape(-1,11, x.size(1)*x.size(2)*x.size(3))
                #print(x.shape)
                x,_= module(x)
        return x
                
    def forward(self,x):
        #print(self.model)
        pred, alphas, temp_alphas = self.model(x)
        #pred = pred.argmax(dim=1)
        print('target retrieving: ', pred[:, 2])
        #pred[:, 2].backward()
        one_hot_output = torch.FloatTensor(288, pred.size()[-1]).zero_()
        one_hot_output[0][2] = 1
        pred.backward(gradient=one_hot_output, retain_graph=True)
        gradients = self.get_activations_gradient()
        print('Gradients retreived: ', gradients.size())    

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        x = self.register_hook(x)
        x = x.view(x.size(0), -1)  # Flatten
        # Forward pass on the classifier
        x = self.model.softmax(x)
        return x
        
class GradCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        #
        conv_output, layer_output = self.extractor.forward_pass_on_convolutions(input_image)
        print(type(self.extractor.gradients))
        model_output, alphas, spat_attn = self.model(input_image)
        #model_output = conv_output
        #print('done here',model_output.size())
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        # Target for backprop
        one_hot_output = torch.FloatTensor(40, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1 ## choosing the target class as one 
        print('target',  model_output.size()[-1])
        # Zero grads
        self.model.zero_grad()
        one_hot_output = one_hot_output#.cuda()
        #self.model.classifier.zero_grad()
        # Backward pass with specified target
        
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients
        print('backkward gradient')
        print(type(self.extractor.gradients))
        guided_gradients = self.extractor.gradients.data
        print('GUIDED: ', guided_gradients.size())
        # Get convolution outputs
        target = conv_output.data[0]
       
        print('target: ', target.size())
        # Get weights from gradients
        #weights = torch.mean(guided_gradients, dim=(1, 2))  # Take averages for each gradient
        guided_gradients = guided_gradients.detach().cpu().numpy()
        print('GUIDED: ', guided_gradients.shape)
        weights = np.mean(guided_gradients, axis=(0)) 
        print('weights: ',weights.shape)
        # Create empty numpy array for cam
        #cam = np.ones(target.shape[1:], dtype=np.float32)
        cam = Variable(torch.ones((target.shape[1:])))
        cam = cam.numpy()
        #weights = weights.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        # Multiply each weight with its conv output and then, sum
        print('target',target.shape)
        all_cams = np.ones(target.shape)
        print('cams',all_cams.shape)
        weights = np.clip(weights, 0, 1000) 
        for i, w in enumerate(weights):
            cam += w * target[0, i, :]
            all_cams[0] = w * target[0, i, :]
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam)) 
        
        all_cams = np.maximum(all_cams, 0)
        all_cams = (all_cams - np.min(all_cams)) / (np.max(all_cams) - np.min(all_cams))
        
        # Normalize between 0-1
        #cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        #cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                       #input_image.shape[3]), Image.ANTIALIAS))/255
        # ^ I am extremely unhappy with this line. Originally resizing was done in cv2 which
        # supports resizing numpy matrices with antialiasing, however,
        # when I moved the repository to PIL, this option was out of the window.
        # So, in order to use resizing with ANTIALIAS feature of PIL,
        # I briefly convert matrix to PIL image and then back.
        # If there is a more beautiful way, do not hesitate to send a PR.
        return cam,all_cams

"""
if __name__ == '__main__':
    # Get params
    target_example = 0  # Snake
    (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
        get_example_params(target_example)
    # Grad cam
    grad_cam = GradCam(pretrained_model, target_layer=11)
    # Generate cam mask
    cam = grad_cam.generate_cam(prep_img, target_class)
    # Save mask
    save_class_activation_images(original_image, cam, file_name_to_export)
    print('Grad cam completed')
    """