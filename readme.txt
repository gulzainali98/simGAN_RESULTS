*************************Directories**********************************

nyu_synth_test: Images from test folder provided in NYU Dataset
nyu_refined: Results of processing images provided in nyu_synth_test folder through the trained generator. 
seen_synth_test: Images on which generator was trained on
seen_refine: Results of processing images provided in seen_refine folder through trained generator. 

**********************************************************************

synth_creator.py: Preprocessing script I used on images from test folder provided in NYU Dataset to convert those images into
                  grayscale with range [0,255]. Script provided by NYU for preprocessing returns normalized images in range [-1,1],
                  i use following formula to scale the range of these images from [-1,1] to [0,255] -> (img+1)*127.5. This returns 
                  image with white hand and black background. To further convert the image to have black hand and white background, i
                  simply subtract image by 255. Hence, final formula to get desired image becomes 255-((img+1)*127.5).

