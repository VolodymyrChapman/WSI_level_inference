from torch import nn, argmax, from_numpy
import pandas as pd
import numpy as np
from utils.preprocessing_wsi import retrieve_patches
import os

# # check that sufficient numbers of patches exist
# patch_size = (224, 224)
# req_mag = 20
# positive_threshold = 1
# num_patches = 'all'
# min_patch_number = 100
# cache_dir = os.path.join(outdir, 'cache')
# feature_batch_size = 100

# feature_extractor = ibot_vit_base_pancan
# feature_extractor_transforms = ibot_vit_base_pancan.transform
# feature_extractor_device = 'cuda'

# scorer_model = model['model']
# scorer_model_device = 'cpu'

def predict_over_wsi(outdir:str, 
                    wsi_patch_size: tuple,
                    req_mag:int,
                    feature_extractor,
                    feature_extractor_transforms,
                    feature_extractor_device:str,
                    scorer_model,
                    scorer_model_device:str,
                    wsi_dataloader, 
                    postive_tissue_threshold: float = 1.0,
                    num_patches = 'all',
                    min_patch_number: int = 100,
                    feature_batch_size: int = 100,
                    cache_dir: str = 'cache'):
                
    """ For prediction over a whole slide image using independent feature extraction and 
    feature scoring models. This is conducted separately for the purposes of efficiency
    when handling models that may have different hardware requirements between feature
    extraction and scoring i.e. Phikon iBotViT on a cuda GPU followed by a smaller,
    lower demand Chowder model on CPU,
        """

    for wsi_files in wsi_dataloader:
        for wsi in wsi_files:
            try:
                wsi = wsi_files[0]
                print('processing',wsi)
                
                outfile = os.path.split(wsi)[-1]
                outfile = os.path.splitext(outfile)[0] + '.tsv'
                outfile = os.path.join(outdir, outfile)

            #     # if inference hasn't previously been conducted
                if os.path.exists(outfile) == False:
                    
                    # retrieve patches into cache storage
                    retrieve_patches(wsi, 
                                outdir = outdir,
                                patch_size = wsi_patch_size,
                                req_mag = req_mag,
                                positive_threshold = positive_tissue_threshold,
                                num_patches = num_patches,
                                transform=transform, 
                                min_patch_number = min_patch_number)
                    
                    patches = os.listdir(cache_dir)

                    if len(patches) > 0:               

                        batches = [patches[i*feature_batch_size:(i+1)*feature_batch_size] for i in range(np.ceil(len(patches)/feature_batch_size).astype(int))]

                        # extract features with requested extractor
                        with torch.no_grad():
                            outputs = []
                            for batch in tqdm(batches):
                                images = [Image.open(os.path.join(cache_dir, file)) for file in batch]
                                images = torch.stack([feature_extractor_transforms(image) for image in images])
                                inputs = images.to(feature_extractor_device) 
                                
                                # feature extraction
                                feature_extractor.to(feature_extractor_device)
                                features = feature_extractor(inputs).detach().cpu()
                                
                                # score features with MIL model - i.e. Chowder
                                # by default, conducted on cpu
                                scorer_model = scorer_model.to(scorer_model_device).eval()
                                scores = scorer_model(features.to(scorer_model_device)).detach().cpu()
                                cols = [f'out_{i}' for i in range(scores.shape[1])]
                                
                                # print(scores_out[0].shape)

                                outputs.append(torch.concat(scores, dim = -1))

                                # clean up RAM
                                if device == 'cpu':
                                    continue
                                else:
                                    with torch.cuda.device(device):
                                        torch.cuda.empty_cache()

                            output_concat = torch.concat(outputs, dim = 0)

                            # prepare output df
                            df = pd.DataFrame(output_concat, columns = cols)
                            df['image_id'] = patches
                            df['start_coord_0'] = [int(id.split('[')[1].split(',')[0]) for id in df['image_id']]
                            df['start_coord_1'] = [int(id.split(']')[0].split(' ')[1]) for id in df['image_id']]
                            df['patch_size'] = [tuple(eval(id.split(')')[0].split('(')[1])) for id in df['image_id']]
                            
                            # clean up memory
                            del patches, batches, inputs, outputs
                            shutil.rmtree(os.path.join(outdir, 'cache'))

                            # save results for slide
                            df.to_csv(outfile, index = False, sep = '\t')

                    else:
                        print(wsi, 'has less than', min_patch_number,'patches -- skipping')

                else:
                    print(wsi, 'inference file exists:', outfile)

            except:
                print('issue with', wsi)
                shutil.rmtree(os.path.join(outdir, 'cache'))
                gc.collect()

            # clean up -  delete cache dir
            gc.collect()


# function for creating heatmaps of whole slide inferences 
def wsi_heatmap(wsi_dataloader,
                scores_df,
                col_to_show:str,
                file_col: str,
                outdir:str, 
                coord_col_1: str = 'start_coord_0',
                coord_col_2: str = 'start_coord_1',
                patch_size_col:str = 'patch_size',
                display_mag:int = 2,
                cmap:str = 'viridis'):

    for batch in wsi_dataloader:
        for filepath in batch:
            wsi_id = os.path.split(filepath[0])[-1]
            wsi_id = str(os.path.splitext(wsi_id)[0])

            # gauge which classification group wsi falls into - TP, TN etc.
            filt_data = scores_df[scores_df[file_col] == wsi_id]

            if len(filt_data) > 0:
                sld = OpenSlide(filepath)
                # determine level of downsample needed for target magnificaiton
                mag_diff = int(sld.properties['aperio.AppMag']) // display_mag
                req_dims = [dim//mag_diff for dim in sld.dimensions]
                req_mag_thumb = np.array(sld.get_thumbnail([dim//mag_diff for dim in sld.dimensions]))

                # np dims reverse of PIL
                heatmap = np.zeros(req_dims[::-1])
                
                filt_patch_df = patch_info_df[scores_df[file_col].astype(str) == wsi_id]
                
                # retrieve coordinates for patches to extract
                coords = list(zip(filt_patch_df[coord_col_2], filt_patch_df[coord_col_1]))
                # for each set of coordinates, add heatmap output
                for patch in range(len(filt_patch_df)):
                    coord = coords[patch]
                    coords_down = [c//mag_diff for c in coord]
                    patch_size_tuple = tuple(eval(filt_patch_df.iloc[patch][patch_size_col]))[::-1]
                    patch_size_down = [s//mag_diff for s in patch_size_tuple]
                    heatmap[coords_down[0]:coords_down[0] + patch_size_down[0], coords_down[1]: coords_down[1] + patch_size_down[1]] = filt_patch_df.iloc[patch][1]
                    
                # create heatmap figure
                fig, ax = plt.subplots(1,1,figsize = (20,20))
                ax.imshow(req_mag_thumb)
                sns.heatmap(heatmap, cmap = cmap,vmin = -0.5, vmax = 0.5, mask = abs(heatmap) < 0.2, ax = ax, alpha = 0.3)
                ax.axis('off')
                
                fig.tight_layout()

                # save figure
                outfile = os.path.join(outdir, f'{wsi_id}_{col_to_show}.tiff')
                plt.savefig(outfile)

                # clean up
                plt.clf() 
                plt.close()
            



















## Function for whole slide inference from patch inferences 

# def wsi_score(scores_df,
                # scores_cols: List = None,
#               wsi_score_model,
#               wsi_score_model_device:str = 'cpu', 
#               wsi_result_file: str = None): 

#     # # add whole slide predictions to predictions output
#     # # infer
#     # softmax = nn.Softmax(dim = 0)
#     # wsi_scores = []

#     # cols = [col for col in df if model['model_file'] in col]
#     # in_data = from_numpy(df[cols].values).transpose(-1, -2)
#     # wsi_model = model['predict_from_scores'].eval()
#     # raw_output = wsi_model(in_data).squeeze().detach()
#     # soft_output = softmax(raw_output)
#     # wsi_scores.append({'model_file':model['model_file'], 
#     # 'wsi':os.path.split(wsi)[-1], 
#     # 'raw_output' : raw_output.numpy(),
#     # 'softmax_out' : soft_output.numpy(),
#     # 'pred_class': argmax(raw_output).numpy()})

#     # # create wsi result file if does not exist
#     # if os.path.exists(wsi_result_file) != True:
#     #     print('Creating WSI results file:', wsi_result_file)
#     #     pd.DataFrame(wsi_scores).to_csv(wsi_result_file, sep = '\t', index = False)
    
#     # else:
#     #     # append to existing predictions file
#     #     class_df = pd.read_csv(wsi_result_file, sep = '\t')
#     #     class_df = pd.concat([class_df, pd.DataFrame(wsi_scores)], axis = 0)
#     #     class_df.to_csv(wsi_result_file, sep = '\t', index = False)
    
#     #     # clean up
#     #     del class_df
    # del wsi_scores, in_data, raw_output, soft_output, softmax
