import os
import numpy as np
import torch
from PIL import Image
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
from imgaug import augmenters as iaa
import imgaug as ia
from scipy.ndimage.measurements import label
from scipy.ndimage.morphology import binary_dilation

# preprocesses ground-truth labelmap (data expected to already provide tubules border as label 7)
def preprocessingGT(lbl):
    structure = np.zeros((3, 3), dtype=np.int)
    structure[1, :] = 1
    structure[:, 1] = 1

    # add glomeruli border only for almost touching glomeruli
    allGlomeruli = np.logical_or(lbl == 2, lbl == 3)
    labeledGlom, numberGlom = label(np.asarray(allGlomeruli, np.uint8), structure)
    temp = np.zeros(lbl.shape)
    for i in range(1, numberGlom + 1):
        temp += binary_dilation(binary_dilation(binary_dilation(binary_dilation(binary_dilation(binary_dilation(binary_dilation(labeledGlom == i)))))))
    glomBorder = np.logical_and(temp > 1, np.logical_not(allGlomeruli))
    lbl[binary_dilation(glomBorder)] = 7

    # add arterial border only for almost touching arteries
    allArteries = np.logical_or(lbl == 5, lbl == 6)
    labeledGlom, numberGlom = label(np.asarray(allArteries, np.uint8), structure)
    temp = np.zeros(lbl.shape)
    for i in range(1, numberGlom + 1):
        temp += binary_dilation(binary_dilation(binary_dilation(binary_dilation(binary_dilation(labeledGlom == i)))))
    glomBorder = np.logical_and(temp > 1, np.logical_not(allArteries))
    lbl[binary_dilation(glomBorder)] = 7

# Class representing either train, val or test dataset
class CustomDataSetRAM(Dataset):
    def __init__(self, datasetType, logger):
        self.transformIMG = None
        self.transformLBL = None
        self.transform_WhenNoAugm = transforms.Compose([
            RangeNormaliziation(),
            ToTensor()
        ])

        self.data = []
        self.useAugm = datasetType=='train'
        #self.useAugm = False
        self.lblShape = 0


        if self.useAugm:
            self.transformIMG, self.transformLBL = get_Augmentation_Transf()
            logger.info('Augmentation method:')
            logger.info(self.transformIMG)

        assert datasetType in ['train', 'val', 'test'], '### ERROR: WRONG DATASET TYPE '+datasetType+' ! ###'

        # please enter path to data folder
        #image_dir_base = '/home/students/ying/jupyter_code/Unet_test/Unet_dataset_TrainValTest'
        #Huge dataset: 516_640_Healthy_UUO_Adenine_Alport_IRI_NTN_8labels_fewHuman_TrainValTest
        #image_dir_base = '/home/students/ying/jupyter_code/Unet_test/516_640_Healthy_UUO_Adenine_Alport_IRI_NTN_8labels_fewHuman_TrainValTest'
        #image_dir_base = '/home/students/ying/jupyter_code/Unet_test/UTNet_dataset_test/Fold_2'
        #image_dir_base = '/home/students/ying/jupyter_code/Unet_test/Masterthesis_Data'
        image_dir_base = '/home/students/ying/jupyter_code/Unet_test/Masterthesis_DataNor'
        image_dir_aug = '/images/DigitalPathology/DataSets/StainAugmentation/AFOG_Fake'
        image_dir_aug2 = '/images/DigitalPathology/DataSets/StainAugmentation/aSMA_Fake'
        image_dir_3 = "/images/DigitalPathology/DataSets/StainAugmentation/CD31_Fake"
        image_dir_4 = "/images/DigitalPathology/DataSets/StainAugmentation/F4-80_Fake"
        #image_dir_aug = '/home/students/ying/jupyter_code/Unet_test/Unet_dataset_TrainValTest'
        #image_dir_base = "/work/scratch/ying/DatasetForMICCAL/Dataset7"


        if datasetType == 'train':
            image_dir = image_dir_base + '/Train'
            image_aug = image_dir_aug + '/Train'
            image_aug2 = image_dir_aug2 + '/Train'
            image_aug3 = image_dir_3 + '/Train'
            image_aug4 = image_dir_4 + '/Train'
        elif datasetType == 'val':
            image_dir = image_dir_base + '/Val'
            image_aug = image_dir_aug + '/Val'
            image_aug2 = image_dir_aug2 + '/Val'
            image_aug3 = image_dir_3 + '/Val'
            image_aug4 = image_dir_4 + '/Val'
        elif datasetType == 'test':
            image_dir = image_dir_base + '/Test'
            image_aug = image_dir_aug + '/Test'
            image_aug2 = image_dir_aug2 + '/Test'
            image_aug3 = image_dir_3 + '/Test'
            image_aug4 = image_dir_4 + '/Test'

        # here we expect labels to be stored in same directory as respective images with ending '-labels.png' instead of '.png'
        label_dir = image_dir
        files = sorted(list(filter(lambda x: ').png' in x, os.listdir(image_dir))))

        logger.info('Loading dataset with size: '+str(len(files)))
        for k, fname in enumerate(files):
            imagePath = os.path.join(image_dir, fname)
            labelPath = os.path.join(label_dir, fname.replace('.png', '-labels.png'))

            #augPath = os.path.join(image_aug, fname.replace('.png', '_fake_AFOG.png'))
            #augPath2 = os.path.join(image_aug2, fname.replace('.png', '_fake_aSMA.png'))
            #augPath3 = os.path.join(image_aug3, fname.replace('.png', '_fake_CD31.png'))
            
            if datasetType == 'train':
                #augPath = os.path.join(image_aug, fname.replace('.png', '_fake_AFOG.png'))
                augPath2 = os.path.join(image_aug2, fname.replace('.png', '_fake_aSMA.png'))
                augPath3 = os.path.join(image_aug3, fname.replace('.png', '_fake_CD31.png'))
                augPath4 = os.path.join(image_aug4, fname.replace('.png', '_fake_F4-80.png'))
                
            elif datasetType == 'val':
                augPath = os.path.join(image_dir, fname)
                augPath2 = os.path.join(image_dir, fname)
                augPath3 = os.path.join(image_dir, fname)
                augPath4 = os.path.join(image_dir, fname)
            elif datasetType == 'test':
                augPath = os.path.join(image_dir, fname)
                augPath2 = os.path.join(image_dir, fname)
                augPath3 = os.path.join(image_dir, fname)
                augPath4 = os.path.join(image_dir, fname)
            
            #augPath = os.path.join(image_aug, fname)

            img = cv2.cvtColor(cv2.imread(imagePath), cv2.COLOR_BGR2RGB)
            #img = cv2.cvtColor(cv2.imread(augPath3), cv2.COLOR_BGR2RGB)
            #img_aug2 = cv2.cvtColor(cv2.imread(augPath2), cv2.COLOR_BGR2RGB)
            #img_aug3 = cv2.cvtColor(cv2.imread(augPath3), cv2.COLOR_BGR2RGB)
            #img_aug4 = cv2.cvtColor(cv2.imread(augPath4), cv2.COLOR_BGR2RGB)

            #Imgresize = transforms.Resize(512, 512)
            #img = img[62:578, 62:578, :]
            img = img[:640, :640, :]
            #img_aug = img_aug[:640, :640, :]
            #img_aug2 = img_aug2[:640, :640, :]
            #img_aug3 = img_aug3[:640, :640, :]
            #img_aug4 = img_aug4[:640, :640, :]
            #img = cv2.resize(img, (258, 258), interpolation=cv2.INTER_AREA)             #test for utnet
            img = cv2.resize(img, (320, 320), interpolation=cv2.INTER_AREA)
            #img_aug = cv2.resize(img_aug, (320, 320), interpolation=cv2.INTER_AREA)
            #img_aug2 = cv2.resize(img_aug2, (320, 320), interpolation=cv2.INTER_AREA)
            #img_aug3 = cv2.resize(img_aug3, (320, 320), interpolation=cv2.INTER_AREA)
            #img_aug4 = cv2.resize(img_aug4, (320, 320), interpolation=cv2.INTER_AREA)
            #img = np.concatenate((img, img_aug), axis=2)
            #img = np.concatenate((img, img_aug2), axis=2)
            #lbl = np.array(Image.open(labelPath))
            lbl = np.array(Image.open(labelPath).resize((258, 258), Image.ANTIALIAS))   #test for utnet


            #In label, border is signed as label=8, try to transform from 8 to 7
            #Be careful about the order of [j][i]


            for i in range(0, lbl.shape[0]):
                for j in range(0, lbl.shape[1]):
                    if lbl[j][i] == 7:
                        lbl[j][i] = 8
                    elif lbl[j][i] == 8:
                        lbl[j][i] = 7
                    else:
                        lbl[j][i] = lbl[j][i]


            # preprocess ground truth
            preprocessingGT(lbl)

            logger.info("Load data with index " + str(k) + " : " + fname + ", ImgShape: " + str(img.shape) + " " + str(img.dtype) + ", LabelShape: " + str(lbl.shape) + " " + str(lbl.dtype) + " (max: " + str(lbl.max()) + ", min: " + str(lbl.min()) + ")")
            
            self.lblShape = lbl.shape
            # most likely, shapes are not equal, then pad label map to same size as images with values of 8 (those values will be ignored for loss computation), providing equal sizes simplifies the appliacation of data augmentation transformation 
            if img.shape[:2] != lbl.shape:
                lbl = np.pad(lbl, ((img.shape[0]-lbl.shape[0])//2,(img.shape[1]-lbl.shape[1])//2), 'constant', constant_values=(8,8))


            #self.data.append((img, lbl))
            
            if self.useAugm:
                for i in range(3):
                    self.data.append((img, lbl))
            else:
                self.data.append((img, lbl))
            
            
            """
            if datasetType == 'train':
                self.data.append((img, lbl))
                self.data.append((img_aug2, lbl))
                self.data.append((img_aug3, lbl))
                self.data.append((img_aug4, lbl))
            else:
                self.data.append((img, lbl))
            """
            '''
            
            self.lblShape = (258, 258)
            for a in range(3):
                for b in range(3):
                    if a == 0 and b == 0:
                        img1 = img[a*160+31:a*160+320+31, b*160+31:b*160+320+31, :]
                        lbl1 = lbl[a*129:a*129+258, b*129:b*129+258]
                        lbl1 = np.pad(lbl1, ((img1.shape[0]-lbl1.shape[0])//2,(img1.shape[1]-lbl1.shape[1])//2), 'constant', constant_values=(8,8))
                        self.data.append((img1, lbl1))
                    elif a == 0 and b == 1:
                        img2 = img[a*160+31:a*160+320+31, b*160:b*160+320, :]
                        lbl2 = lbl[a*129:a*129+258, b*129:b*129+258]
                        lbl2 = np.pad(lbl2, ((img2.shape[0]-lbl2.shape[0])//2,(img2.shape[1]-lbl2.shape[1])//2), 'constant', constant_values=(8,8))
                        self.data.append((img2, lbl2))
                    elif a == 0 and b == 2:
                        img3 = img[a*160+31:a*160+320+31, b*160-31:b*160+320-31, :]
                        lbl3 = lbl[a*129:a*129+258, b*129:b*129+258]
                        lbl3 = np.pad(lbl3, ((img3.shape[0]-lbl3.shape[0])//2,(img3.shape[1]-lbl3.shape[1])//2), 'constant', constant_values=(8,8))
                        self.data.append((img3, lbl3))
                    elif a == 1 and b == 0:
                        img4 = img[a*160:a*160+320, b*160+31:b*160+320+31, :]
                        lbl4 = lbl[a*129:a*129+258, b*129:b*129+258]
                        lbl4 = np.pad(lbl4, ((img4.shape[0]-lbl4.shape[0])//2,(img4.shape[1]-lbl4.shape[1])//2), 'constant', constant_values=(8,8))
                        self.data.append((img4, lbl4))
                    elif a == 1 and b == 1:
                        img5 = img[a*160:a*160+320, b*160:b*160+320, :]
                        lbl5 = lbl[a*129:a*129+258, b*129:b*129+258]
                        lbl5 = np.pad(lbl5, ((img5.shape[0]-lbl5.shape[0])//2,(img5.shape[1]-lbl5.shape[1])//2), 'constant', constant_values=(8,8))
                        self.data.append((img5, lbl5))
                    elif a == 1 and b == 2:
                        img6 = img[a*160:a*160+320, b*160-31:b*160+320-31, :]
                        lbl6 = lbl[a*129:a*129+258, b*129:b*129+258]
                        lbl6 = np.pad(lbl6, ((img6.shape[0]-lbl6.shape[0])//2,(img6.shape[1]-lbl6.shape[1])//2), 'constant', constant_values=(8,8))
                        self.data.append((img6, lbl6))
                    elif a == 2 and b == 0:
                        img7 = img[a*160-31:a*160+320-31, b*160+31:b*160+320+31, :]
                        lbl7 = lbl[a*129:a*129+258, b*129:b*129+258]
                        lbl7 = np.pad(lbl7, ((img7.shape[0]-lbl7.shape[0])//2,(img7.shape[1]-lbl7.shape[1])//2), 'constant', constant_values=(8,8))
                        self.data.append((img7, lbl7))
                    elif a == 2 and b == 1:
                        img8 = img[a*160-31:a*160+320-31, b*160:b*160+320, :]
                        lbl8 = lbl[a*129:a*129+258, b*129:b*129+258]
                        lbl8 = np.pad(lbl8, ((img8.shape[0]-lbl8.shape[0])//2,(img8.shape[1]-lbl8.shape[1])//2), 'constant', constant_values=(8,8))
                        self.data.append((img8, lbl8))
                    else:
                        img9 = img[a*160-31:a*160+320-31, b*160-31:b*160+320-31, :]
                        lbl9 = lbl[a*129:a*129+258, b*129:b*129+258]
                        lbl9 = np.pad(lbl9, ((img9.shape[0]-lbl9.shape[0])//2,(img9.shape[1]-lbl9.shape[1])//2), 'constant', constant_values=(8,8))
                        self.data.append((img9, lbl9))
            '''
            

        assert len(files) > 0, 'No files found in ' + image_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.useAugm:
            # get different augmentation transformation for each sample within minibatch
            ia.seed(np.random.get_state()[1][0])

            img, lbl = self.data[index]

            seq_img_d = self.transformIMG.to_deterministic()
            seq_lbl_d = self.transformLBL.to_deterministic()

            # apply almost equal transformation for label maps (however using nearest neighbor interpolation)
            seq_lbl_d = seq_lbl_d.copy_random_state(seq_img_d, matching="name")

            # after applying the transformation, center crop label map back to its original size
            augmentedIMG = seq_img_d.augment_image(img)
            augmentedLBL = seq_lbl_d.augment_image(lbl)[(img.shape[0]-self.lblShape[0])//2:(img.shape[0]-self.lblShape[0])//2+self.lblShape[0],(img.shape[1]-self.lblShape[1])//2:(img.shape[1]-self.lblShape[1])//2+self.lblShape[1]]

            return self.transform_WhenNoAugm((augmentedIMG, augmentedLBL.copy()))
        else:
            img, lbl = self.data[index]
            return self.transform_WhenNoAugm((img, lbl[(img.shape[0]-self.lblShape[0])//2:(img.shape[0]-self.lblShape[0])//2+self.lblShape[0],(img.shape[1]-self.lblShape[1])//2:(img.shape[1]-self.lblShape[1])//2+self.lblShape[1]]))

# normalize images to interval [-1.6, 1.6]
class RangeNormaliziation(object):
    def __call__(self, sample):
        img, lbl = sample
        return img / 255.0 * 3.2 - 1.6, lbl
        #return img / 255.0 * 3.2*4 - 1.6*4, lbl        #test

    
class ToTensor(object):
    def __call__(self, sample):
        img, lbl = sample

        lbl = torch.from_numpy(lbl).long()
        img = torch.from_numpy(np.array(img, np.float32).transpose(2, 0, 1))

        return img, lbl



def get_Augmentation_Transf():
    # 'images' should be either a 4D numpy array of shape (N, height, width, channels)
    # or a list of 3D numpy arrays, each having shape (height, width, channels).
    # Grayscale images must have shape (height, width, 1) each.
    # All images must have numpy's dtype uint8. Values are expected to be in
    # range 0-255.
    sometimes = lambda aug: iaa.Sometimes(0.5, aug, name="Random1")
    sometimes2 = lambda aug: iaa.Sometimes(0.2, aug, name="Random2")
    sometimes3 = lambda aug: iaa.Sometimes(0.9, aug, name="Random3")
    sometimes4 = lambda aug: iaa.Sometimes(0.9, aug, name="Random4")
    sometimes5 = lambda aug: iaa.Sometimes(0.9, aug, name="Random5")

    # specify DATA AUGMENTATION TRANSFORMATION
    seq_img = iaa.Sequential([
        iaa.AddToHueAndSaturation(value=(-13, 13), name="MyHSV"),
        sometimes2(iaa.GammaContrast(gamma=(0.85, 1.15), name="MyGamma")),
        iaa.Fliplr(0.5, name="MyFlipLR"),
        iaa.Flipud(0.5, name="MyFlipUD"),
        #iaa.Rotate((-30, 30), name="MyRot"),
        sometimes(iaa.Rot90(k=1, keep_size=True, name="MyRot90")),
        iaa.OneOf([
            sometimes3(iaa.PiecewiseAffine(scale=(0.015, 0.02), cval=0, name="MyPiece")),
            sometimes4(iaa.ElasticTransformation(alpha=(100, 200), sigma=20, cval=0, name="MyElastic")),
            sometimes5(iaa.Affine(scale={"x": (0.95, 1.05), "y": (0.95, 1.05)}, rotate=(-45, 45), shear=(-4, 4), cval=0, name="MyAffine"))
        ], name="MyOneOf")
    ], name="MyAug")

    seq_lbl = iaa.Sequential([
        iaa.Fliplr(0.5, name="MyFlipLR"),
        iaa.Flipud(0.5, name="MyFlipUD"),
        #iaa.Rotate((-30, 30), name="MyRot"),
        sometimes(iaa.Rot90(k=1, keep_size=True, name="MyRot90")),
        iaa.OneOf([
            sometimes3(iaa.PiecewiseAffine(scale=(0.015, 0.02), cval=8, order=0, name="MyPiece")),
            sometimes4(iaa.ElasticTransformation(alpha=(100, 200), sigma=20, cval=8, order=0, name="MyElastic")),
            sometimes5(iaa.Affine(scale={"x": (0.95, 1.05), "y": (0.95, 1.05)}, rotate=(-45, 45), shear=(-4, 4), cval=8, order=0, name="MyAffine"))
        ], name="MyOneOf")
    ], name="MyAug")

    return seq_img, seq_lbl



if '__main__' == __name__:
    print()
