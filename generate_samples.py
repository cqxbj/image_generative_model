import functions as my_F

# generate _conditional_residual_attention_DDPM.pth
model_name = "_conditional_residual_attention_DDPM"
save_path =  "_FID_model_gen_images"
model, diffuser = my_F.load_ddpm_model(model_name, is_attention_on=True, is_residual_on=True,n_label=10)
# my_F.generate_FID_samples(model, diffuser, path=model_name, conditional= True)
my_F.generate_gird_imgs(model, diffuser, n_label=10, modelname=model_name, epoch_index= 999)

# FID
# then run python -m pytorch_fid _conditional_residual_attention_DDPM cifar_real_images