from django.shortcuts import render,redirect
from .models import *
#from django.contrib.auth.models import User,auth
from django.core.files.storage import FileSystemStorage

import os
import random
import torch
from PIL import Image
from django.conf import settings


from torchvision import transforms
from .gan_module import Generator

model = Generator(ngf=32, n_residual_blocks=9)
ckpt = torch.load('pretrained_model/state_dict.pth', map_location='cpu')
model.load_state_dict(ckpt)
model.eval()

trans = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])


#def index (request):
#    return render (request,'index.html')
#def login(request):
#    if request.method=='POST':
#        UserName=request.POST['user_name']
#        Password=request.POST['password']   
#        if UserReg.objects.filter(user__username=UserName).exists():
#            u=auth.authenticate(username=UserName,password=Password)
#            if u is not None:
#                auth.login(request,u)
#                return redirect(userhome)
#            else:
#                context = {
#                    'key':'invalid user'
#                }
#                return render(request,'login.html',context)
#        else:
#            context = {
#                'key':'invalid user'
#            }
#            return render(request,'login.html',context)    
#    return render(request,'login.html')

#def reg(request):
#    if request.method=='POST':
#        name=request.POST['name']
#        email=request.POST['email']
#        address=request.POST['address']
#        phone_no=request.POST['phone_no']
#        password=request.POST['password']
#        confirm_password=request.POST['confirm_password']
        
#        if password==confirm_password:
#            if UserReg.objects.filter(user__username=name).exists():
#                context = {
#                'msg':'user already exists....'
#                }
#                return render(request,'reg.html',context)
#            else:
#                User.objects.create_user(username=name,email=email,password=password).save()
#                user=User.objects.get(username=name)
#                UserReg(user=user,address=address,phone_no=phone_no).save()
                # context = {
                # 'msg':'successfully registered.'
                # }
#                return redirect(login)
#        else:
#            context = {
#                'msg':'password doesnot match....'
#            }
#            return render(request,'reg.html',context)
#    return render(request,'reg.html')
def userhome(request):
    if request.method == 'POST' and request.FILES.get('fileUpload'):
        uploaded_file = request.FILES['fileUpload']
        fs = FileSystemStorage(location=r'app\static\media')
        print(fs)
        filename = fs.save(uploaded_file.name,uploaded_file)
        print(filename)

        media = settings.MEDIA_ROOT
        img=os.path.join(media,filename)
        
        img = Image.open(img).convert('RGB')
        img = trans(img).unsqueeze(0)
        aged_face = model(img)
        aged_face = (aged_face.squeeze().permute(1, 2, 0).detach().numpy() + 1.0) / 2.0


        output_path = os.path.join(media,'output.png')
        aged_face_pil = Image.fromarray((aged_face * 255).astype('uint8'))
        aged_face_pil.save(output_path)
        print(img)
        
       

        context={
            'key':fs.url(output_path),
            'key2':fs.url(filename),
        }
        return render (request,'result.html',context)
    return render (request,'user_home.html')


   
