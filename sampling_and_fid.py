import functions as my_F


# generate _conditional_residual_attention_DDPM.pth
# model_name = "_DDPM"
# save_path =  "_FID_model_gen_images_ddpm"
# model, diffuser = my_F.load_ddpm_model(model_name, is_attention_on=False, is_residual_on=False,n_label=10)
# my_F.generate_imgs(model, diffuser, save_folder=save_path, n_label=10, n_100= 50)
# my_F.generate_gird_imgs(model, diffuser, n_label=10, modelname=model_name, epoch_index= 999)



if __name__ == '__main__':
    ddpm_path =  "_FID_model_gen_images_ddpm"
    fid_1  = my_F.calcualte_fid(ddpm_path, "cifar_test_real_images")
    print("fid_normal: ", fid_1)
    c_ddpm_path =  "_FID_model_gen_images"
    fid_2  = my_F.calcualte_fid(c_ddpm_path, "cifar_test_real_images")
    print("fid_advanced_model", fid_2)
