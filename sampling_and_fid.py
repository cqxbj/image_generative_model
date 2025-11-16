import functions as my_F



# model_name = "_conditional_residual_attention_DDPM"
# save_path =  "_FID_ddpm_model_advanced"
# model, diffuser = my_F.load_ddpm_model(model_name, is_attention_on=True, is_residual_on=True,n_label=10)
# my_F.generate_imgs(model, diffuser, save_folder=save_path, n_label=10, n_100= 50)
# my_F.generate_gird_imgs(model, diffuser, n_label=10, modelname=model_name, epoch_index= 999)

# model_name = "vae"
# model = my_F.load_vae_model(model_name)
# my_F.vae_generate_imgs(model)




test_Str = """COMP7015 Artificial Intelligence – Group Project Instructions
1. Overall Requirements
1. Groups: Form a group of 1 to 5 students. Forming groups with students from other 
sections is allowed. (At least 3 members in a team are recommended)
2. Milestones:
o Group Registration & Topic Selection: Due by 11:59 pm, 24th October 2025. 
Please register your group members and chosen topic via the following link: 
https://hkbuchtl.qualtrics.com/jfe/form/SV_eS9K0CSvz9A9JvU.
o Final Submission: Due by 11:59 pm, 21st November 2025. This includes your 
source code and project report. One submission for each team will be enough.
o In-person Presentation: Scheduled for 22nd and 23rd November 2025. A detailed 
schedule will be announced after the group registration deadline.
3. Final Submission Package:
o Project Report: A PDF document of at most five A4 pages (single column). It 
should describe your project's motivation, methods, results, and a discussion. A 
mandatory section must detail the contribution of each group member.
o Source Codes: A single .zip file. Your code will be evaluated in the FSC 8/F lab 
environment, so ensure it runs smoothly there. Acknowledge all major third-party 
libraries (e.g., PyTorch, TensorFlow, Hugging Face Transformers) in your report.
4. Presentation: Each group will have approximately 8 minutes for their presentation, 
followed by a Q&A session. Every member must present their part of the work. The use 
of visualizations (figures, graphs, demos) is highly encouraged to clearly convey your 
project's story.



Topic 3: Open Topic
If you are interested in another AI problem that aligns with the course content, 
you may propose your own project. 
This is an opportunity to explore areas like generative AI, multimodal learning, 
foundation models, or other advanced deep learning applications.
You are free to choose any relevant dataset and AI/deep learning methods.
Your project should demonstrate a clear understanding of the principles and techniques taught in this course.
Remember that the chosen topic's difficulty, scope, and creativity will be key factors in your evaluation.


Evaluation Criteria
Your project will be evaluated based on a holistic assessment of your work across several 
dimensions:
• Project Completeness & Model Performance: How thoroughly the project tasks were 
completed and the effectiveness of your final model(s) on the chosen task.
• Creativity & Difficulty: The novelty of your approach, the technical challenge of the 
problem, and the sophistication of the methods used.
• Code Quality: The readability, organization, and documentation of your source code. 
Clean and well-structured code is expected.
• Storytelling (Report & Presentation): The clarity and depth of your project report and 
presentation. This includes how well you explain your motivation, describe your 
methodology, analyze your results, and present your conclusions.
""".upper()

print(test_Str)


model_name = "gans"
generator, discriminator = my_F.load_gans_model(model_name)

my_F.gans_save_samples(generator, epoch=-1, input_str=test_Str)
# my_F.gans_generate_imgs(generator)

''' 

    this .py is implemented for calculating FID scores.

'''

# if __name__ == '__main__':
#     fid_1  = my_F.calcualte_fid("hw_train", "gans")
#     print(fid_1)
# my_F.save_hand_writing_images("hw_train")