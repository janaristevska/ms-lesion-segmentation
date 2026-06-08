import os
import numpy as np
from nibabel import load as load_nii
import nibabel as nib
from operator import itemgetter
from .build_model import define_training_layers, fit_model
from operator import add


def train_cascaded_model(model, train_x_data, train_y_data, options):
    """
    Train the model using a cascade of two CNN
    """
    # ---------- CNN1 ----------
    print("> CNN: loading training data for first model")
    X, Y, sel_voxels = load_training_data(train_x_data, train_y_data, options)
    print('> CNN: train_x ', X.shape)

    if options['full_train'] is False:
        max_epochs = options['max_epochs']
        patience = 0
        best_val_loss = np.Inf
        model[0] = define_training_layers(model=model[0],
                                          num_layers=options['num_layers'],
                                          number_of_samples=X.shape[0])
        options['max_epochs'] = 0
        for it in range(0, max_epochs, 10):
            options['max_epochs'] += 10
            model[0] = fit_model(model[0], X, Y, options,
                                 initial_epoch=it)

            # evaluate if continuing training or not
            val_loss = min(model[0]['history'].history['val_loss'])
            if val_loss > best_val_loss:
                patience += 10
            else:
                best_val_loss = val_loss

            if patience >= options['patience']:
                break

            X, Y, sel_voxels = load_training_data(train_x_data,
                                                  train_y_data,
                                                  options)
        options['max_epochs'] = max_epochs
    else:
        model[0] = fit_model(model[0], X, Y, options)

    # ---------- CNN2 ----------
    print('> CNN: loading training data for the second model')
    X, Y, sel_voxels = load_training_data(train_x_data,
                                          train_y_data,
                                          options,
                                          model=model[0])
    print('> CNN: train_x ', X.shape)

    if options['full_train'] is False:
        max_epochs = options['max_epochs']
        patience = 0
        best_val_loss = np.Inf
        model[1] = define_training_layers(model=model[1],
                                          num_layers=options['num_layers'],
                                          number_of_samples=X.shape[0])

        options['max_epochs'] = 0
        for it in range(0, max_epochs, 10):
            options['max_epochs'] += 10
            model[1] = fit_model(model[1], X, Y, options,
                                 initial_epoch=it)

            # evaluate if continuing training or not
            val_loss = min(model[0]['history'].history['val_loss'])
            if val_loss > best_val_loss:
                patience += 10
            else:
                best_val_loss = val_loss

            if patience >= options['patience']:
                break

            X, Y, sel_voxels = load_training_data(train_x_data,
                                                  train_y_data,
                                                  options,
                                                  model=model[0],
                                                  selected_voxels=sel_voxels)
        options['max_epochs'] = max_epochs
    else:
        model[1] = fit_model(model[1], X, Y, options)

    return model


def test_cascaded_model(model, test_x_data, options):
    """
    Test the cascaded approach using a learned model
    """
    exp_folder = os.path.join(options['test_folder'],
                              options['test_scan'],
                              options['experiment'])
    if not os.path.exists(exp_folder):
        os.mkdir(exp_folder)

    options['test_name'] = options['experiment'] + '_debug_prob_0.nii.gz'
    save_nifti = True if options['debug'] is True else False
    t1 = test_scan(model[0], test_x_data, options, save_nifti=save_nifti)

    t1 = t1 > 0.8
    if np.sum(t1) > 0:
        options['test_name'] = options['experiment'] + '_prob_1.nii.gz'
        t2 = test_scan(model[1],
                       test_x_data,
                       options,
                       save_nifti=True,
                       candidate_mask=(t1))
    else:
        t2 = np.zeros(t1.shape)

    # FIX: Convert keys to list for indexing
    scans = list(test_x_data.keys())
    flair_scans = [test_x_data[s]['FLAIR'] for s in scans]
    flair_image = load_nii(flair_scans[0])
    options['test_name'] = options['experiment'] + '_hard_seg.nii.gz'
    out_segmentation = post_process_segmentation(t2,
                                                 options,
                                                 save_nifti=True,
                                                 orientation=flair_image.affine)
    return out_segmentation


def load_training_data(train_x_data,
                       train_y_data,
                       options,
                       model=None,
                       selected_voxels=None):
    '''
    Load training and label samples for all given scans and modalities.
    '''
    scans = list(train_x_data.keys())
    modalities = list(train_x_data[scans[0]].keys())

    if model is None:
        flair_scans = [train_x_data[s]['FLAIR'] for s in scans]
        selected_voxels = select_training_voxels(flair_scans,
                                                 options['min_th'])
    elif selected_voxels is None:
        selected_voxels = select_voxels_from_previous_model(model,
                                                            train_x_data,
                                                            options)
    else:
          pass

    data = []
    for m in modalities:
        x_data = [train_x_data[s][m] for s in scans]
        y_data = [train_y_data[s] for s in scans]
        x_patches, y_patches = load_train_patches(x_data,
                                                  y_data,
                                                  selected_voxels,
                                                  options['patch_size'],
                                                  options['balanced_training'],
                                                  options['fract_negative_positive'])
        data.append(x_patches)

    X = np.stack(data, axis=1)
    Y = y_patches

    if options['randomize_train']:
        seed = np.random.randint(np.iinfo(np.int32).max)
        np.random.seed(seed)
        X = np.random.permutation(X.astype(dtype=np.float32))
        np.random.seed(seed)
        Y = np.random.permutation(Y.astype(dtype=np.int32))

    if options['fully_convolutional']:
        Y = np.expand_dims(Y, axis=1)
    else:
        # FIX: Use integer division for indexing
        if Y.shape[3] == 1:
            Y = Y[:, Y.shape[1] // 2, Y.shape[2] // 2, :]
        else:
            Y = Y[:, Y.shape[1] // 2, Y.shape[2] // 2, Y.shape[3] // 2]
        Y = np.squeeze(Y)

    return X, Y, selected_voxels


def normalize_data(im, datatype=np.float32):
    """
    zero mean / 1 standard deviation image normalization
    """
    im = im.astype(dtype=datatype) - im[np.nonzero(im)].mean()
    im = im / im[np.nonzero(im)].std()
    return im


def select_training_voxels(input_masks, threshold=2, datatype=np.float32):
    """
    Select voxels for training based on a intensity threshold
    """
    images = [load_nii(image_name).get_data() for image_name in input_masks]
    images_norm = [normalize_data(im) for im in images]
    rois = [image > threshold for image in images_norm]
    return rois


def load_train_patches(x_data,
                       y_data,
                       selected_voxels,
                       patch_size,
                       balanced_training,
                       fraction_negatives,
                       random_state=42,
                       datatype=np.float32):
    """
    Load train patches
    """
    images = [load_nii(name).get_data() for name in x_data]
    images_norm = [normalize_data(im) for im in images]

    lesion_masks = [load_nii(name).get_data().astype(bool) for name in y_data]
    nolesion_masks = [np.logical_and(np.logical_not(lesion), brain)
                      for lesion, brain in zip(lesion_masks, selected_voxels)]

    lesion_centers = [get_mask_voxels(mask) for mask in lesion_masks]
    nolesion_centers = [get_mask_voxels(mask) for mask in nolesion_masks]

    np.random.seed(random_state)

    number_lesions = [np.sum(lesion) for lesion in lesion_masks]
    total_lesions = np.sum(number_lesions)
    neg_samples = int((total_lesions * fraction_negatives) / len(number_lesions))
    X, Y = [], []

    for l_centers, nl_centers, image, lesion in zip(lesion_centers,
                                                    nolesion_centers,
                                                    images_norm,
                                                    lesion_masks):
        if balanced_training:
            if len(l_centers) > 0:
                x_pos_samples = get_patches(image, l_centers, patch_size)
                y_pos_samples = get_patches(lesion, l_centers, patch_size)
                idx = np.random.permutation(range(0, len(nl_centers))).tolist()[:len(l_centers)]
                nolesion = itemgetter(*idx)(nl_centers)
                x_neg_samples = get_patches(image, nolesion, patch_size)
                y_neg_samples = get_patches(lesion, nolesion, patch_size)
                X.append(np.concatenate([x_pos_samples, x_neg_samples]))
                Y.append(np.concatenate([y_pos_samples, y_neg_samples]))
        else:
            if len(l_centers) > 0:
                x_pos_samples = get_patches(image, l_centers, patch_size)
                y_pos_samples = get_patches(lesion, l_centers, patch_size)

            idx = np.random.permutation(range(0, len(nl_centers))).tolist()[:neg_samples]
            nolesion = itemgetter(*idx)(nl_centers)
            x_neg_samples = get_patches(image, nolesion, patch_size)
            y_neg_samples = get_patches(lesion, nolesion, patch_size)

            if len(l_centers) > 0:
                X.append(np.concatenate([x_pos_samples, x_neg_samples]))
                Y.append(np.concatenate([y_pos_samples, y_neg_samples]))
            else:
                X.append(x_neg_samples)
                Y.append(y_neg_samples)

    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)
    return X, Y


def load_test_patches(test_x_data,
                      patch_size,
                      batch_size,
                      voxel_candidates=None,
                      datatype=np.float32):
    """
    Function generator to load test patches
    """
    # FIX: Convert keys to list for indexing
    scans = list(test_x_data.keys())
    modalities = list(test_x_data[scans[0]].keys())

    images = []
    for m in modalities:
        raw_images = [load_nii(test_x_data[s][m]).get_data() for s in scans]
        images.append([normalize_data(im) for im in raw_images])

    if voxel_candidates is None:
        flair_scans = [test_x_data[s]['FLAIR'] for s in scans]
        selected_voxels = [get_mask_voxels(mask)
                           for mask in select_training_voxels(flair_scans, 0.5)][0]
    else:
        selected_voxels = get_mask_voxels(voxel_candidates)

    for i in range(0, len(selected_voxels), batch_size):
        c_centers = selected_voxels[i:i + batch_size]
        X = []
        for m, image_modality in zip(modalities, images):
            X.append(get_patches(image_modality[0], c_centers, patch_size))
        yield np.stack(X, axis=1), c_centers


def get_mask_voxels(mask):
    """
    Compute x,y,z coordinates of a binary mask
    """
    indices = np.stack(np.nonzero(mask), axis=1)
    indices = [tuple(idx) for idx in indices]
    return indices


def get_patches(image, centers, patch_size=(15, 15, 15)):
    """
    Get image patches of arbitrary size based on a set of centers
    """
    patches = []
    list_of_tuples = all([isinstance(center, tuple) for center in centers])
    sizes_match = [len(center) == len(patch_size) for center in centers]

    if list_of_tuples and sizes_match:
        patch_half = tuple([idx // 2 for idx in patch_size])
        # FIX: Convert map object to list/tuple
        new_centers = [tuple(map(add, center, patch_half)) for center in centers]
        padding = tuple((idx, size - idx)
                        for idx, size in zip(patch_half, patch_size))
        new_image = np.pad(image, padding, mode='constant', constant_values=0)
        slices = [[slice(c_idx - p_idx, c_idx + (s_idx - p_idx))
                   for (c_idx, p_idx, s_idx) in zip(center,
                                                    patch_half,
                                                    patch_size)]
                  for center in new_centers]
        # FIX: Use tuple for multidimensional indexing
        patches = [new_image[tuple(idx)] for idx in slices]

    return patches


def test_scan(model,
              test_x_data,
              options,
              save_nifti=True,
              candidate_mask=None):
    """
    Test data based on one model
    """
    # FIX: Convert keys to list for indexing
    scans = list(test_x_data.keys())
    flair_scans = [test_x_data[s]['FLAIR'] for s in scans]
    flair_image = load_nii(flair_scans[0])
    seg_image = np.zeros_like(flair_image.get_data().astype('float32'))

    if candidate_mask is not None:
        all_voxels = np.sum(candidate_mask)
    else:
        all_voxels = np.sum(flair_image.get_data() > 0)

    if options['debug'] is True:
        print("> DEBUG ", scans[0], "Voxels to classify:", all_voxels)

    for batch, centers in load_test_patches(test_x_data,
                                            options['patch_size'],
                                            options['batch_size'],
                                            candidate_mask):
        if options['debug'] is True:
            print("> DEBUG: testing current_batch:", batch.shape)

        y_pred = model['net'].predict(np.squeeze(batch),
                                      options['batch_size'])
        [x, y, z] = np.stack(centers, axis=1)
        seg_image[x, y, z] = y_pred[:, 1]
    if options['debug'] is True:
        print("...done!")

    if check_min_error(seg_image, options, flair_image.header.get_zooms()):
        if options['debug']:
            print("> DEBUG ", scans[0], "lesion volume below ", \
                  options['min_error'], 'ml')
        seg_image = np.zeros_like(flair_image.get_data().astype('float32'))

    if save_nifti:
        out_scan = nib.Nifti1Image(seg_image, affine=flair_image.affine)
        out_scan.to_filename(os.path.join(options['test_folder'],
                                          options['test_scan'],
                                          options['experiment'],
                                          options['test_name']))

    return seg_image


def check_min_error(input_scan, options, voxel_size):
    """
    check that the output volume is higher than the minimum accuracy
    """
    from scipy import ndimage

    t_bin = options['t_bin']
    l_min = options['l_min']
    voxel_size = np.prod(voxel_size) / 1000.0

    output_scan = np.zeros_like(input_scan)
    t_segmentation = input_scan >= t_bin

    labels, num_labels = ndimage.label(t_segmentation)
    if num_labels == 0:
        return (np.sum(output_scan == 1) * voxel_size) < options['min_error']

    label_list = np.unique(labels)
    num_elements_by_lesion = ndimage.labeled_comprehension(t_segmentation,
                                                           labels,
                                                           label_list,
                                                           np.sum,
                                                           float, 0)

    for l, _ in enumerate(num_elements_by_lesion):
        if num_elements_by_lesion[l] > l_min:
            current_voxels = np.stack(np.where(labels == l), axis=1)
            output_scan[current_voxels[:, 0],
            current_voxels[:, 1],
            current_voxels[:, 2]] = 1

    return (np.sum(output_scan == 1) * voxel_size) < options['min_error']


def select_voxels_from_previous_model(model, train_x_data, options):
    """
    Select training voxels from image segmentation masks
    """
    scans = list(train_x_data.keys())
    seg_masks = []
    for scan, s in zip(scans, range(len(scans))):
        seg_mask = test_scan(model,
                             dict(list(train_x_data.items())[s:s + 1]),
                             options, save_nifti=False)
        seg_masks.append(seg_mask > 0.5)

        if options['debug']:
            flair = nib.load(train_x_data[scan]['FLAIR'])
            tmp_seg = nib.Nifti1Image(seg_mask,
                                      affine=flair.affine)
            tmp_seg.to_filename(os.path.join(options['weight_paths'],
                                             options['experiment'],
                                             '.train',
                                             scan + '_it0.nii.gz'))

    flair_scans = [train_x_data[s]['FLAIR'] for s in scans]
    images = [load_nii(name).get_data() for name in flair_scans]
    images_norm = [normalize_data(im) for im in images]

    seg_mask = [im > 2 if np.sum(seg) == 0 else seg
                for im, seg in zip(images_norm, seg_masks)]

    return seg_mask


def post_process_segmentation(input_scan,
                              options,
                              save_nifti=True,
                              orientation=np.eye(4)):
    """
    Post-process the probabilistic segmentation
    """
    from scipy import ndimage

    t_bin = options['t_bin']
    l_min = options['l_min']
    output_scan = np.zeros_like(input_scan)
    t_segmentation = input_scan >= t_bin

    labels, num_labels = ndimage.label(t_segmentation)
    if num_labels == 0:
        return output_scan

    label_list = np.unique(labels)
    num_elements_by_lesion = ndimage.labeled_comprehension(t_segmentation,
                                                           labels,
                                                           label_list,
                                                           np.sum,
                                                           float, 0)

    for l in range(len(num_elements_by_lesion)):
        if num_elements_by_lesion[l] > l_min:
            current_voxels = np.stack(np.where(labels == l), axis=1)
            output_scan[current_voxels[:, 0],
            current_voxels[:, 1],
            current_voxels[:, 2]] = 1

    if save_nifti:
        nifti_out = nib.Nifti1Image(output_scan,
                                    affine=orientation)
        nifti_out.to_filename(os.path.join(options['test_folder'],
                                           options['test_scan'],
                                           options['experiment'],
                                           options['test_name']))

    return output_scan




# # # PROMENA NA KODOT ZA DA RABOTAM SO PRVIOT MODEL
# #
# import os
# import numpy as np
# from nibabel import load as load_nii
# import nibabel as nib
# from operator import itemgetter
# from .build_model import define_training_layers, fit_model
# from operator import add
#
#
# def train_cascaded_model(model, train_x_data, train_y_data, options):
#     """
#     Train the model using a cascade of two CNN
#     """
#     # ---------- CNN1 ----------
#     print("> CNN: loading training data for first model")
#     X, Y, sel_voxels = load_training_data(train_x_data, train_y_data, options)
#     print('> CNN: train_x ', X.shape)
#
#     if options['full_train'] is False:
#         max_epochs = options['max_epochs']
#         patience = 0
#         best_val_loss = np.Inf
#         model[0] = define_training_layers(model=model[0],
#                                           num_layers=options['num_layers'],
#                                           number_of_samples=X.shape[0])
#         options['max_epochs'] = 0
#         for it in range(0, max_epochs, 10):
#             options['max_epochs'] += 10
#             model[0] = fit_model(model[0], X, Y, options,
#                                  initial_epoch=it)
#
#             # evaluate if continuing training or not
#             val_loss = min(model[0]['history'].history['val_loss'])
#             if val_loss > best_val_loss:
#                 patience += 10
#             else:
#                 best_val_loss = val_loss
#
#             if patience >= options['patience']:
#                 break
#
#             X, Y, sel_voxels = load_training_data(train_x_data,
#                                                   train_y_data,
#                                                   options)
#         options['max_epochs'] = max_epochs
#     else:
#         model[0] = fit_model(model[0], X, Y, options)
#
#     # ---------- CNN2 ----------
#     print('> CNN: loading training data for the second model')
#     X, Y, sel_voxels = load_training_data(train_x_data,
#                                           train_y_data,
#                                           options,
#                                           model=model[0])
#     print('> CNN: train_x ', X.shape)
#
#     if options['full_train'] is False:
#         max_epochs = options['max_epochs']
#         patience = 0
#         best_val_loss = np.Inf
#         model[1] = define_training_layers(model=model[1],
#                                           num_layers=options['num_layers'],
#                                           number_of_samples=X.shape[0])
#
#         options['max_epochs'] = 0
#         for it in range(0, max_epochs, 10):
#             options['max_epochs'] += 10
#             model[1] = fit_model(model[1], X, Y, options,
#                                  initial_epoch=it)
#
#             # evaluate if continuing training or not
#             val_loss = min(model[0]['history'].history['val_loss'])
#             if val_loss > best_val_loss:
#                 patience += 10
#             else:
#                 best_val_loss = val_loss
#
#             if patience >= options['patience']:
#                 break
#
#             X, Y, sel_voxels = load_training_data(train_x_data,
#                                                   train_y_data,
#                                                   options,
#                                                   model=model[0],
#                                                   selected_voxels=sel_voxels)
#         options['max_epochs'] = max_epochs
#     else:
#         model[1] = fit_model(model[1], X, Y, options)
#
#     return model
#
#
# def test_cascaded_model(model, test_x_data, options):
#     """
#     Test the cascaded approach using one or two CNNs
#     """
#     exp_folder = os.path.join(options['test_folder'],
#                               options['test_scan'],
#                               options['experiment'])
#     if not os.path.exists(exp_folder):
#         os.mkdir(exp_folder)
#
#     options['test_name'] = options['experiment'] + '_debug_prob_0.nii.gz'
#     save_nifti = True if options['debug'] is True else False
#     t1 = test_scan(model[0], test_x_data, options, save_nifti=save_nifti)
#
#     # If cascade is disabled or only one model exists, skip second CNN
#     if (not options.get('cascaded', True)) or len(model) == 1:
#         t2 = t1
#     else:
#         t1 = t1 > 0.8
#         if np.sum(t1) > 0:
#             options['test_name'] = options['experiment'] + '_prob_1.nii.gz'
#             t2 = test_scan(model[1],
#                            test_x_data,
#                            options,
#                            save_nifti=True,
#                            candidate_mask=(t1))
#         else:
#             t2 = np.zeros(t1.shape)
#
#     # Save final hard segmentation
#     scans = list(test_x_data.keys())
#     flair_scans = [test_x_data[s]['FLAIR'] for s in scans]
#     flair_image = load_nii(flair_scans[0])
#     options['test_name'] = options['experiment'] + '_hard_seg.nii.gz'
#     out_segmentation = post_process_segmentation(t2,
#                                                  options,
#                                                  save_nifti=True,
#                                                  orientation=flair_image.affine)
#     return out_segmentation
#
#
#
# def load_training_data(train_x_data,
#                        train_y_data,
#                        options,
#                        model=None,
#                        selected_voxels=None):
#     '''
#     Load training and label samples for all given scans and modalities.
#     '''
#     scans = list(train_x_data.keys())
#     modalities = list(train_x_data[scans[0]].keys())
#
#     if model is None:
#         flair_scans = [train_x_data[s]['FLAIR'] for s in scans]
#         selected_voxels = select_training_voxels(flair_scans,
#                                                  options['min_th'])
#     elif selected_voxels is None:
#         selected_voxels = select_voxels_from_previous_model(model,
#                                                             train_x_data,
#                                                             options)
#
#     data = []
#     for m in modalities:
#         x_data = [train_x_data[s][m] for s in scans]
#         y_data = [train_y_data[s] for s in scans]
#         x_patches, y_patches = load_train_patches(x_data,
#                                                   y_data,
#                                                   selected_voxels,
#                                                   options['patch_size'],
#                                                   options['balanced_training'],
#                                                   options['fract_negative_positive'])
#         data.append(x_patches)
#
#     X = np.stack(data, axis=1)
#     Y = y_patches
#
#     if options['randomize_train']:
#         seed = np.random.randint(np.iinfo(np.int32).max)
#         np.random.seed(seed)
#         X = np.random.permutation(X.astype(dtype=np.float32))
#         np.random.seed(seed)
#         Y = np.random.permutation(Y.astype(dtype=np.int32))
#
#     if options['fully_convolutional']:
#         Y = np.expand_dims(Y, axis=1)
#     else:
#         # FIX: Use integer division for indexing
#         if Y.shape[3] == 1:
#             Y = Y[:, Y.shape[1] // 2, Y.shape[2] // 2, :]
#         else:
#             Y = Y[:, Y.shape[1] // 2, Y.shape[2] // 2, Y.shape[3] // 2]
#         Y = np.squeeze(Y)
#
#     return X, Y, selected_voxels
#
#
# def normalize_data(im, datatype=np.float32):
#     """
#     zero mean / 1 standard deviation image normalization
#     """
#     im = im.astype(dtype=datatype) - im[np.nonzero(im)].mean()
#     im = im / im[np.nonzero(im)].std()
#     return im
#
#
# def select_training_voxels(input_masks, threshold=2, datatype=np.float32):
#     """
#     Select voxels for training based on a intensity threshold
#     """
#     images = [load_nii(image_name).get_data() for image_name in input_masks]
#     images_norm = [normalize_data(im) for im in images]
#     rois = [image > threshold for image in images_norm]
#     return rois
#
#
# def load_train_patches(x_data,
#                        y_data,
#                        selected_voxels,
#                        patch_size,
#                        balanced_training,
#                        fraction_negatives,
#                        random_state=42,
#                        datatype=np.float32):
#     """
#     Load train patches
#     """
#     images = [load_nii(name).get_data() for name in x_data]
#     images_norm = [normalize_data(im) for im in images]
#
#     lesion_masks = [load_nii(name).get_data().astype(bool) for name in y_data]
#     nolesion_masks = [np.logical_and(np.logical_not(lesion), brain)
#                       for lesion, brain in zip(lesion_masks, selected_voxels)]
#
#     lesion_centers = [get_mask_voxels(mask) for mask in lesion_masks]
#     nolesion_centers = [get_mask_voxels(mask) for mask in nolesion_masks]
#
#     np.random.seed(random_state)
#
#     number_lesions = [np.sum(lesion) for lesion in lesion_masks]
#     total_lesions = np.sum(number_lesions)
#     neg_samples = int((total_lesions * fraction_negatives) / len(number_lesions))
#     X, Y = [], []
#
#     for l_centers, nl_centers, image, lesion in zip(lesion_centers,
#                                                     nolesion_centers,
#                                                     images_norm,
#                                                     lesion_masks):
#         if balanced_training:
#             if len(l_centers) > 0:
#                 x_pos_samples = get_patches(image, l_centers, patch_size)
#                 y_pos_samples = get_patches(lesion, l_centers, patch_size)
#                 idx = np.random.permutation(range(0, len(nl_centers))).tolist()[:len(l_centers)]
#                 nolesion = itemgetter(*idx)(nl_centers)
#                 x_neg_samples = get_patches(image, nolesion, patch_size)
#                 y_neg_samples = get_patches(lesion, nolesion, patch_size)
#                 X.append(np.concatenate([x_pos_samples, x_neg_samples]))
#                 Y.append(np.concatenate([y_pos_samples, y_neg_samples]))
#         else:
#             if len(l_centers) > 0:
#                 x_pos_samples = get_patches(image, l_centers, patch_size)
#                 y_pos_samples = get_patches(lesion, l_centers, patch_size)
#
#             idx = np.random.permutation(range(0, len(nl_centers))).tolist()[:neg_samples]
#             nolesion = itemgetter(*idx)(nl_centers)
#             x_neg_samples = get_patches(image, nolesion, patch_size)
#             y_neg_samples = get_patches(lesion, nolesion, patch_size)
#
#             if len(l_centers) > 0:
#                 X.append(np.concatenate([x_pos_samples, x_neg_samples]))
#                 Y.append(np.concatenate([y_pos_samples, y_neg_samples]))
#             else:
#                 X.append(x_neg_samples)
#                 Y.append(y_neg_samples)
#
#     X = np.concatenate(X, axis=0)
#     Y = np.concatenate(Y, axis=0)
#     return X, Y
#
#
# def load_test_patches(test_x_data,
#                       patch_size,
#                       batch_size,
#                       voxel_candidates=None,
#                       datatype=np.float32):
#     """
#     Function generator to load test patches
#     """
#     # FIX: Convert keys to list for indexing
#     scans = list(test_x_data.keys())
#     modalities = list(test_x_data[scans[0]].keys())
#
#     images = []
#     for m in modalities:
#         raw_images = [load_nii(test_x_data[s][m]).get_data() for s in scans]
#         images.append([normalize_data(im) for im in raw_images])
#
#     if voxel_candidates is None:
#         flair_scans = [test_x_data[s]['FLAIR'] for s in scans]
#         selected_voxels = [get_mask_voxels(mask)
#                            for mask in select_training_voxels(flair_scans, 0.5)][0]
#     else:
#         selected_voxels = get_mask_voxels(voxel_candidates)
#
#     for i in range(0, len(selected_voxels), batch_size):
#         c_centers = selected_voxels[i:i + batch_size]
#         X = []
#         for m, image_modality in zip(modalities, images):
#             X.append(get_patches(image_modality[0], c_centers, patch_size))
#         yield np.stack(X, axis=1), c_centers
#
#
# def get_mask_voxels(mask):
#     """
#     Compute x,y,z coordinates of a binary mask
#     """
#     indices = np.stack(np.nonzero(mask), axis=1)
#     indices = [tuple(idx) for idx in indices]
#     return indices
#
#
# def get_patches(image, centers, patch_size=(15, 15, 15)):
#     """
#     Get image patches of arbitrary size based on a set of centers
#     """
#     patches = []
#     list_of_tuples = all([isinstance(center, tuple) for center in centers])
#     sizes_match = [len(center) == len(patch_size) for center in centers]
#
#     if list_of_tuples and sizes_match:
#         patch_half = tuple([idx // 2 for idx in patch_size])
#         # FIX: Convert map object to list/tuple
#         new_centers = [tuple(map(add, center, patch_half)) for center in centers]
#         padding = tuple((idx, size - idx)
#                         for idx, size in zip(patch_half, patch_size))
#         new_image = np.pad(image, padding, mode='constant', constant_values=0)
#         slices = [[slice(c_idx - p_idx, c_idx + (s_idx - p_idx))
#                    for (c_idx, p_idx, s_idx) in zip(center,
#                                                     patch_half,
#                                                     patch_size)]
#                   for center in new_centers]
#         # FIX: Use tuple for multidimensional indexing
#         patches = [new_image[tuple(idx)] for idx in slices]
#
#     return patches
#
#
# def test_scan(model,
#               test_x_data,
#               options,
#               save_nifti=True,
#               candidate_mask=None):
#     """
#     Test data based on one model
#     """
#     # FIX: Convert keys to list for indexing
#     scans = list(test_x_data.keys())
#     flair_scans = [test_x_data[s]['FLAIR'] for s in scans]
#     flair_image = load_nii(flair_scans[0])
#     seg_image = np.zeros_like(flair_image.get_data().astype('float32'))
#
#     if candidate_mask is not None:
#         all_voxels = np.sum(candidate_mask)
#     else:
#         all_voxels = np.sum(flair_image.get_data() > 0)
#
#     if options['debug'] is True:
#         print("> DEBUG ", scans[0], "Voxels to classify:", all_voxels)
#
#     for batch, centers in load_test_patches(test_x_data,
#                                             options['patch_size'],
#                                             options['batch_size'],
#                                             candidate_mask):
#         if options['debug'] is True:
#             print("> DEBUG: testing current_batch:", batch.shape)
#
#         y_pred = model['net'].predict(np.squeeze(batch),
#                                       options['batch_size'])
#         [x, y, z] = np.stack(centers, axis=1)
#         seg_image[x, y, z] = y_pred[:, 1]
#     if options['debug'] is True:
#         print("...done!")
#
#     if check_min_error(seg_image, options, flair_image.header.get_zooms()):
#         if options['debug']:
#             print("> DEBUG ", scans[0], "lesion volume below ", \
#                   options['min_error'], 'ml')
#         seg_image = np.zeros_like(flair_image.get_data().astype('float32'))
#
#     if save_nifti:
#         out_scan = nib.Nifti1Image(seg_image, affine=flair_image.affine)
#         out_scan.to_filename(os.path.join(options['test_folder'],
#                                           options['test_scan'],
#                                           options['experiment'],
#                                           options['test_name']))
#
#     return seg_image
#
#
# def check_min_error(input_scan, options, voxel_size):
#     """
#     check that the output volume is higher than the minimum accuracy
#     """
#     from scipy import ndimage
#
#     t_bin = options['t_bin']
#     l_min = options['l_min']
#     voxel_size = np.prod(voxel_size) / 1000.0
#
#     output_scan = np.zeros_like(input_scan)
#     t_segmentation = input_scan >= t_bin
#
#     labels, num_labels = ndimage.label(t_segmentation)
#     if num_labels == 0:
#         return (np.sum(output_scan == 1) * voxel_size) < options['min_error']
#
#     label_list = np.unique(labels)
#     num_elements_by_lesion = ndimage.labeled_comprehension(t_segmentation,
#                                                            labels,
#                                                            label_list,
#                                                            np.sum,
#                                                            float, 0)
#
#     for l, _ in enumerate(num_elements_by_lesion):
#         if num_elements_by_lesion[l] > l_min:
#             current_voxels = np.stack(np.where(labels == l), axis=1)
#             output_scan[current_voxels[:, 0],
#             current_voxels[:, 1],
#             current_voxels[:, 2]] = 1
#
#     return (np.sum(output_scan == 1) * voxel_size) < options['min_error']
#
#
# def select_voxels_from_previous_model(model, train_x_data, options):
#     """
#     Select training voxels from image segmentation masks
#     """
#     scans = list(train_x_data.keys())
#     seg_masks = []
#     for scan, s in zip(scans, range(len(scans))):
#         seg_mask = test_scan(model,
#                              dict(list(train_x_data.items())[s:s + 1]),
#                              options, save_nifti=False)
#         seg_masks.append(seg_mask > 0.5)
#
#         if options['debug']:
#             flair = nib.load(train_x_data[scan]['FLAIR'])
#             tmp_seg = nib.Nifti1Image(seg_mask,
#                                       affine=flair.affine)
#             tmp_seg.to_filename(os.path.join(options['weight_paths'],
#                                              options['experiment'],
#                                              '.train',
#                                              scan + '_it0.nii.gz'))
#
#     flair_scans = [train_x_data[s]['FLAIR'] for s in scans]
#     images = [load_nii(name).get_data() for name in flair_scans]
#     images_norm = [normalize_data(im) for im in images]
#
#     seg_mask = [im > 2 if np.sum(seg) == 0 else seg
#                 for im, seg in zip(images_norm, seg_masks)]
#
#     return seg_mask
#
#
# def post_process_segmentation(input_scan,
#                               options,
#                               save_nifti=True,
#                               orientation=np.eye(4)):
#     """
#     Post-process the probabilistic segmentation
#     """
#     from scipy import ndimage
#
#     t_bin = options['t_bin']
#     l_min = options['l_min']
#     output_scan = np.zeros_like(input_scan)
#     t_segmentation = input_scan >= t_bin
#
#     labels, num_labels = ndimage.label(t_segmentation)
#     if num_labels == 0:
#         return output_scan
#
#     label_list = np.unique(labels)
#     num_elements_by_lesion = ndimage.labeled_comprehension(t_segmentation,
#                                                            labels,
#                                                            label_list,
#                                                            np.sum,
#                                                            float, 0)
#
#     for l in range(len(num_elements_by_lesion)):
#         if num_elements_by_lesion[l] > l_min:
#             current_voxels = np.stack(np.where(labels == l), axis=1)
#             output_scan[current_voxels[:, 0],
#             current_voxels[:, 1],
#             current_voxels[:, 2]] = 1
#
#     if save_nifti:
#         nifti_out = nib.Nifti1Image(output_scan,
#                                     affine=orientation)
#         nifti_out.to_filename(os.path.join(options['test_folder'],
#                                            options['test_scan'],
#                                            options['experiment'],
#                                            options['test_name']))
#
#     return output_scan

