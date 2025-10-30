# Diagnosis-LVO-on-non-contrast-CT
Deep Learning for Automated Large-Vessel Segmentation and Occlusion Localization on Noncontrast Brain CT
The overall workflow of multi-modal segmentation model  
![image](https://github.com/zsunAI/Diagnosis-LVO-on-non-contrast-CT/blob/main/png/Fig2.png)  

Stage1: nnU-Net code for training segmentation models of vessels on NCCT images using nnU-Net framework.
Details of the background and running inference is here (https://github.com/MIC-DKFZ/nnUNet).  
<p align="center">
![image](https://github.com/zsunAI/Diagnosis-LVO-on-non-contrast-CT/blob/main/png/Fig1.png)
</p>  

Stage2: a knowledge-augmented multichannel ResNet-18–based network (McResNet) was developed for detection of LVO.
The model incorporated multiple prior-knowledge inputs, including probabilistic arterial territory maps,
hemispheric difference maps, and vessel segmentation masks obtained from stage 1.  
These inputs were fused into a multichannel framework to enhance the discriminative representation of vascular and perfusion asymmetry patterns. A convolutional block attention module (CBAM) was further embedded to capture both channel- and spatial-level contextual dependencies, improving feature learning for LVO identification.  
/stage2/prior-knowledge  
			|--prob_and_half.py  
			|--get_halfbrain_difference.py  
			|--MNI152_brain.nii.gz  
			|--half_brain_mask.nii.gz  
			|--ProbArterialAtlas_BMM_1_double_prep.nii.gz

**Document Description of .py**：
**prob_and_half.py** Using the registration method, with the help of the MNI atlas and:
1. MNI ArterialAtlas probability atlas
2. MNI corresponding left and right brain masks
we can obtain the stroke probability atlas in individual space as well as the left and right brain images in individual space.
**get_halfbrain_difference.py**
The images of the left and right brains obtained above still require the following operations, which involve more steps than obtaining the stroke probability atlas:
1. Flipping: Mirror one side of the brain to the opposite side.
2. Registration: After flipping, the two hemispheres are chirally symmetric, so direct subtraction will not align properly. Therefore, it is necessary to perform registration again.
3. Denoising, filtering, and interpolation calculations.

**Document Description of nii.gz：**
MNI152_brain.nii.gz：brain tissue in MNI space
half_brain_mask.nii.gz: Masks for the left and right brains in MNI space, with different labels assigned to the left and right brains
ProbArterialAtlas_BMM_1_double_prep.nii.gz：Cerebral infarction probability map/cerebral blood supply map


stage3: 2D MIP images of the segmented large vessels were generated, 
and a 2D nnU-Net model was applied for precise localization and classification of the occlusion site 
(M1, M2, C7, C7+M1, or M1+M2).
