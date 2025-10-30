# Diagnosis-LVO-on-non-contrast-CT
Deep Learning for Automated Large-Vessel Segmentation and Occlusion Localization on Noncontrast Brain CT
![image](https://github.com/zsunAI/Diagnosis-LVO-on-non-contrast-CT/blob/main/png/Fig2.png)  
Fig1. Overview of the three-stage deep learning workflow for automated large vessel occlusion (LVO) detection and localization from non-contrast CT (NCCT) scans. Stage 1: A 3D nnU-Net model was used for cerebral vessel segmentation from original 3D axial NCCT slices. Stage 2: The segmented vessel mask, together with probabilistic arterial territory atlases and hemispheric difference maps, were used as multi-channel inputs to a 3D McResNet model for LVO detection. Stage 3: MIP images of large vessels were generated, and a 2D nnU-Net model was applied for LVO localization and occluded segment classification.  
---

**Stage1: nnU-Net code for training segmentation models of vessels on NCCT images using nnU-Net framework.**
Details of the background and running inference is here (https://github.com/MIC-DKFZ/nnUNet). Input patch size, batch size, and voxel spacing follow the specific configurations defined by the respective nnU-Net plans.
<center>
    <img src="https://github.com/zsunAI/Diagnosis-LVO-on-non-contrast-CT/blob/main/png/Fig1.png" alt="image" />
</center>
Fig2. (a) NCCT image; (b) Multi segmentation labels on NCCT images. Green represents the area of C7, blue indicates the area of M1, yellow corresponds to the area of M2, and red signifies other vessels.  

---
**Stage2: a knowledge-augmented multichannel ResNet-18–based network (McResNet) was developed for detection of LVO.**
The model incorporated multiple prior-knowledge inputs, including probabilistic arterial territory maps,
hemispheric difference maps, and vessel segmentation masks obtained from stage 1.  
These inputs were fused into a multichannel framework to enhance the discriminative representation of vascular and perfusion asymmetry patterns. A convolutional block attention module (CBAM) was further embedded to capture both channel- and spatial-level contextual dependencies, improving feature learning for LVO identification.  
*/stage2/prior-knowledge  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--prob_and_half.py  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--get_halfbrain_difference.py  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--MNI152_brain.nii.gz  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--half_brain_mask.nii.gz  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--ProbArterialAtlas_BMM_1_double_prep.nii.gz*

- 🟢 Document Description of .py：
  prob_and_half.py Using the registration method, with the help of the MNI atlas, MNI ArterialAtlas probability atlas, and MNI corresponding left and right brain masks. We can obtain the stroke probability atlas in individual space as well as the left and right brain images in individual space.
- 🟢 get_halfbrain_difference.py:
  The images of the left and right brains obtained above still require the following operations, which involve more steps than obtaining the stroke probability atlas: 1. Flipping: Mirror one side of the brain to the opposite side. 2. Registration: After flipping, the two hemispheres are chirally symmetric, so direct subtraction will not align properly. Therefore, it is necessary to perform registration again. 3. Denoising, filtering, and interpolation calculations.
- 🟢 Document Description of nii.gz:
- MNI152_brain.nii.gz：brain tissue in MNI space
- half_brain_mask.nii.gz: Masks for the left and right brains in MNI space, with different labels assigned to the left and right brains
- ProbArterialAtlas_BMM_1_double_prep.nii.gz：Cerebral infarction probability map/cerebral blood supply map

---
**stage3: 2D MIP images of the segmented large vessels were generated.**
and a 2D nnU-Net model was applied for precise localization and classification of the occlusion site 
(M1, M2, C7, C7+M1, or M1+M2).
