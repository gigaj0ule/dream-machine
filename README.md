# DREAM MACHINE

Ez-to-use diffusion artmaker for low power graphics cards


# SETUP

$ apt install python3 python3-pip git

$ git clone https://github.com/gigaj0ule/dream-machine.git

$ cd dream-machine

$ pip3 install torch numpy omegaconf tqdm einops torchvision pytorch_lightning pandas transformers taming-transformers-rom1504 scipy clip kornia

$ mkdir ai_models

$ cd ai_models

$ wget http://dream.thotcrime.org:8080/ai_models/[the_model_u_want].ckpt

(visit http://dream.thotcrime.org:8080/ai_models/ to see what models are available)



# USAGE

$ cd (git_directory)

$ python3 ./dream_machine/txt2img.py



# DEMO

$ ssh zbox@dream.thotcrime.org -p 8079 

$ password: recreation 

$ cd ~/Desktop/dream-machine 

$ python3 dream_machine/txt2img.py

(ouptut images: http://dream.thotcrime.org:8080/output_txt2img)
