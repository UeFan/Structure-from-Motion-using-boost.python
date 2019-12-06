import os
import sys
import numpy as np
import cv2
import BundleAdjustment
from skimage import img_as_ubyte

def extract_features(images):
    """
    Args:
        - images: list of string

    Returns:
        - key_points_for_all: list of lists
        - descriptor_for_all: list of 2d ndarray (N x 128)
        - colors_for_all: list of 2d ndarray (N x 3)
    """
    sift = cv2.xfeatures2d.SIFT_create(0, 3, 0.04, 10)

    key_points_for_all = []
    descriptor_for_all = []
    colors_for_all = []

    for i in range(len(images)):
        image = cv2.imread(images[i])
        if image is None:
            print ('Jump: {}'.format(images[i]))
            continue

        print("Extracting features:", images[i])

        key_points, descriptor = sift.detectAndCompute(image, None)

        if len(key_points) <= 10:
            continue

        key_points_for_all.append(key_points)
        descriptor_for_all.append(descriptor)
        key_coords = cv2.KeyPoint_convert(key_points)


        colors = np.zeros((len(key_points), 3))
        colors = image[np.round(key_coords[:, 1]).astype(np.int32), np.round(key_coords[:, 0]).astype(np.int32), :]
        colors_for_all.append(colors)
        print (descriptor)
        print (type(descriptor[0,0]))
        print (descriptor.shape[:2])
    return key_points_for_all, descriptor_for_all, colors_for_all


def match_features_two(des1, des2):
    # there's two definations of this func in C++ with different input
    """
    Args:
        - des1: 2d ndarray of descriptor (N1 x 128)
        - des2: 2d ndarray of descriptor (N2 x 128)

    Returns:
        - matches: 1d list
    """
    #
    # bf = cv2.BFMatcher(cv2.NORM_L2)
    #
    matches = []
    # #
    # # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    #
    # # knn_matches = bf.knnMatch(des1, des2, k = 2)
    #
    # # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    #
    # knn_matches = bf.knnMatch(des1, des2, k=2)
    # min_dist = 0x7fffffff
    # # for r in range(len(knn_matches)):
    # #     if knn_matches[r][0].distance > 0.7 * knn_matches[r][1].distance:
    # #         continue
    # #
    # #     dist = knn_matches[r][0].distance
    # #     if dist < min_dist:
    # #         min_dist = dist
    # for r in range(len(knn_matches)):
    #     if knn_matches[r][0].distance > 0.7 * knn_matches[r][1].distance:
    #             # or knn_matches[r][0].distance > 5 * min_dist :# max(min_dist, 10.):
    #         continue
    #
    #     matches.append(knn_matches[r][0])
    # matches = sorted(matches, key = lambda x:x.distance)
    # matches = matches[:10]
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    knn_matches = flann.knnMatch(des1, des2, k=2)
    # Need to draw only good matches, so create a mask
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(knn_matches):
        if m.distance < 0.7 * n.distance:
            matches.append(knn_matches[i][0])

    # print (matches)
    return matches


def match_features_multi(descriptor_for_all):
    """
    Args:
        - descriptor_for_all: list of 2d ndarray
    Returns:
        - matches_for_all: 2d list
    """
    matches_for_all = []
    for i in range(len(descriptor_for_all) - 1):
        print("Matching images %d - %d" % (i, i + 1))
        matches = match_features_two(descriptor_for_all[i], descriptor_for_all[i + 1])
        matches_for_all.append(matches)
    return matches_for_all


def find_transform(K, p1, p2):
    """
    Args:
        - K: intrinsic matrix of camera
        - p1: 2d point in the 1st view
        - p2: 2d point in the 2nd view
    Returns:
        - valid: bool
        - R: orientation of camera
        - T: translation of camera
        - mask: inliers
    """


    R = None
    T = None
    mask = None
    
    print ("K:{}".format(K))

    focal_length =  0.5*(K[0, 0] + K[1, 1])
    principle_point = (K[0, 2], K[1, 2])


    print ("before: p1:{},\n p2:{}\n, focal_l:{}, principle_point:{}, ran:{}".format(p1, p2, focal_length, principle_point, cv2.RANSAC))
    print ("-----------------")

    E, mask = cv2.findEssentialMat(p1, p2, focal_length, principle_point, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None:
        raise Exception("E is None")
    print ("mask: {}\n E: {}".format(mask,E)) # --

    feasible_count = np.sum(mask)
    print("%d inliers in %d" % (feasible_count, len(mask)))
    if (feasible_count < 5) or (feasible_count / len(mask)) < 0.4:
        raise Exception("too few inliers")


    pass_count, R, T, mask = cv2.recoverPose(E, p1, p2)

    # if (pass_count / feasible_count) < 0.5:
    #     print ('pass_count: {}'.format(pass_count))
    #     raise Exception("too few pass_count")
    return R, -T, mask


def get_matched_points(kp1, kp2, matches):
    """
    Args:
        - kp1: list of KeyPoints
        - kp2: list of KeyPoinst
        - matches: list of DMatch
    Returns:
        - out_kp1: 2d ndarray (coordinates of matched kp1)
        - out_kp2: 2d ndarray (cooridnates of matched kp2)
    """
    query_Idx = [match.queryIdx for match in matches]
    train_Idx = [match.trainIdx for match in matches]
    coords1 = cv2.KeyPoint_convert(kp1)
    coords2 = cv2.KeyPoint_convert(kp2)
    out_kp1 = coords1[query_Idx, :]
    out_kp2 = coords2[train_Idx, :]
    return out_kp1, out_kp2


def get_matched_colors(c1, c2, matches):
    """
    Args:
        - c1: 2d ndarray
        - c2: 2d ndarray
        - matches: list of DMatch
    Returns:
        - out_c1: 2d ndarray
        - out_c2: 2d ndarray
    """
    out_c1 = [c1[match.queryIdx, :] for match in matches]
    out_c2 = [c2[match.trainIdx, :] for match in matches]
    return np.array(out_c1), np.array(out_c2)


def reconstruct(K, R1, T1, R2, T2, p1, p2):
    proj1 = np.zeros((3, 4))
    proj2 = np.zeros((3, 4))

    proj1[0:3, 0:3] = R1
    proj1[:, 3] = T1.reshape(proj1.shape[0],)

    proj2[0:3, 0:3] = R2
    proj2[:, 3] = T2.reshape(proj2.shape[0],)

    proj1 = K @ proj1
    proj2 = K @ proj2
    p1_array = np.zeros((2,len(p1)), dtype = np.double)
    p2_array = np.zeros((2,len(p2)), dtype = np.double)
    for i in range(len(p1)):
        p1_array[:,i] = p1[i]
    for i in range(len(p2)):
        p2_array[:,i] = p2[i]
    p_4d = cv2.triangulatePoints(proj1, proj2, p1_array, p2_array)
    structure = p_4d[0:3, :] / p_4d[3, :]
    structure = structure.T
    return structure


def maskout_points(p1, mask):
    p1_after_maskout = []

    for i in range(len(mask)):
        if mask[i] > 0:
            p1_after_maskout.append(p1[i])

    return p1_after_maskout


def maskout_colors(colors, mask):

    colors_after_maskout = []

    for i in range(len(mask)):
        if mask[i] > 0:
            colors_after_maskout.append(colors[i])

    return colors_after_maskout


def get_objpoints_and_imgpoints(matches, structure_indices, structure, key_points):
    object_points = np.empty([0,3])
    image_points = np.empty([0,2])
    print ("The num of match {}".format(len(matches)))
    for i in range(len(matches)):
        query_idx = matches[i].queryIdx
        train_idx = matches[i].trainIdx

        struct_idx = structure_indices[query_idx]
        if struct_idx < 0:
            continue
        object_points = np.vstack((object_points, structure[struct_idx]))
        image_points = np.vstack((image_points, key_points[train_idx].pt))
    return object_points, image_points


def fusion_structure(matches, struct_indices, next_struct_indices, structure, next_structure, colors, next_colors):
    for i in range(len(matches)):
        query_idx = matches[i].queryIdx
        train_idx = matches[i].trainIdx

        struct_idx = struct_indices[query_idx]
        if struct_idx >= 0:
            next_struct_indices[train_idx] = struct_idx
            continue

        structure = np.vstack((structure, next_structure[i]))
        colors.append(next_colors[i])
        struct_indices[query_idx] = next_struct_indices[train_idx] = len(structure) - 1

    return structure, colors, struct_indices, next_struct_indices


def init_structure(K, key_points_for_all, colors_for_all, matches_for_all):
    p1, p2 = get_matched_points(key_points_for_all[0], key_points_for_all[1], matches_for_all[0])

    colors, c2 = get_matched_colors(colors_for_all[0], colors_for_all[1], matches_for_all[0])
    R, T, mask = find_transform(K, p1, p2)

    p1 = maskout_points(p1, mask)
    p2 = maskout_points(p2, mask)
    colors = maskout_colors(colors, mask)
    print ("Init_structure\nnum of p1: {}, num of p2: {}".format(len(p1),len(p2)))
    print ('K: {}, \n p1: {} \n p2: {} \n T:{} \n R:{}'.format(K, p1, p2, T, R))  # --


    R0 = np.eye(3, dtype=np.double)
    T0 = np.zeros((3,1), dtype=np.double)

    structure = reconstruct(K, R0, T0, R, T, p1, p2)

    rotations = []
    rotations.append(R0)
    rotations.append(R)
    motions = []
    motions.append(T0)
    motions.append(T)

    correspond_struct_idx = []
    for i in range(len(key_points_for_all)):
        num = len(key_points_for_all[i])
        correspond_struct_idx.append(np.full((num), -1))

    idx = 0
    matches = matches_for_all[0]
    for i in range(len(matches)):
        if mask[i] == 0:
            continue
        correspond_struct_idx[0][matches[i].queryIdx] = idx
        correspond_struct_idx[1][matches[i].trainIdx] = idx
        idx += 1

    return structure, correspond_struct_idx, colors, rotations, motions


# A struct for BA named ReprojectCost


# def bundle_adjustment():


def main():
    data_path = "/Users/fanyue/Downloads/sfm_images/"
    img_names = os.listdir(data_path)
    file_names = []
    img_path = []
    for i in range(len(img_names)):
        if (img_names[i].split('.')[1] == 'png'):
            file_names.append(img_names[i])
    file_names = sorted(file_names, key=lambda x: (ord(x[7]) * 100 + ord(x[8]) * 10 + ord(x[9])))
    for i in range(len(file_names)):
        img_path.append(data_path + file_names[i] )

    print (img_path)
    K =   np.array([[1520.4, 0, 302.32],[0, 1525.9, 246.87],[0, 0, 1]], dtype = np.float32) # camera matrix

    #np.array([[3310.4, 0, 316.73], [0, 3325.5, 200.55], [0, 0, 1]], dtype = np.float32)#
    ef = BundleAdjustment.EF(img_path)
    key_points_for_all__ = ef.return_kp_list_to_py()
    key_points_for_all = []

    for kps in key_points_for_all__:
        i = 0
        # key_points_for_all_array = np.empty((0, 2))
        kkp = []
        for kp in kps:
            i += 1
            # key_points_for_all_array = np.vstack( (key_points_for_all_array,  ))
            kkp.append(cv2.KeyPoint(kp[0], kp[1], kp[2]))
        key_points_for_all.append(kkp)
        # print ("new kp {}".format(key_points_for_all_array))
        print ("kp:{}".format(i))

    colors_for_all__ = ef.return_color_list_to_py()
    colors_for_all = []


    for color in colors_for_all__:
        i = 0

        colors_for_all_array = np.empty((0, 3), dtype = np.uint8)
        for co in color:
            i += 1
            c = img_as_ubyte(np.array(([co[0], co[1], co[2]]) ))
            colors_for_all_array = np.vstack( (colors_for_all_array,  c))
        colors_for_all.append(colors_for_all_array)
        print ("new color {}".format(i))

    descriptor_for_all__ = ef.return_des_list_to_py()
    descriptor_for_all = []

    for descriptor in descriptor_for_all__:
        # descriptor_for_all_array = np.empty((0, len(descriptor[0])))

        descriptor_for_all_array = np.array(np.float32(descriptor))
        descriptor_for_all.append(descriptor_for_all_array)
        print ("new des {}".format(descriptor_for_all_array))

    # print ("old desc")
    # key_points_for_all, descriptor_for_all, colors_for_all = extract_features(img_path)
    # print ("old des {}".format(descriptor_for_all_array[0].shape[:2]))

    matches_for_all = match_features_multi(descriptor_for_all)

    structure, correspond_struct_idx, colors, rotations, motions = init_structure(K, key_points_for_all, colors_for_all,
                                                                                  matches_for_all)
    print ("start loo")

    # print (matches_for_all)
    for i in range(1, len(matches_for_all)):
        object_points, image_points = get_objpoints_and_imgpoints(matches_for_all[i], correspond_struct_idx[i],
                                                                  structure, key_points_for_all[i + 1])
        dist = np.zeros((1,5), dtype=np.float32)
        print ("{},\n object_points: {},\n image_points: {}".format(i, object_points, image_points))
        if(len(object_points)<4):
            print ("too few object points, continue")
            continue
        _, r, T, inliers = cv2.solvePnPRansac(object_points, image_points, K, dist) # Not sure
        R,_ = cv2.Rodrigues(r)
        rotations.append(R)
        motions.append(T)
        print ('T[{}]: {}'.format(i+1, T)) # --

        p1, p2 = get_matched_points(key_points_for_all[i], key_points_for_all[i + 1], matches_for_all[i])
        c1, c2 = get_matched_colors(colors_for_all[i], colors_for_all[i + 1], matches_for_all[i])

        next_structure = reconstruct(K, rotations[i], motions[i], R, T, p1, p2)

        structure, colors, correspond_struct_idx[i], correspond_struct_idx[i + 1] = fusion_structure(matches_for_all[i], correspond_struct_idx[i], correspond_struct_idx[i + 1],
                                                                                    structure, next_structure, colors, c1)

    intrinsic = []
    intrinsic.append(np.array([[K[0, 0], K[1, 1], K[0, 2], K[1, 2]]], dtype=np.double))
    extrinsics = []
    for i in range(len(rotations)):
        e = np.zeros((6,1), dtype=np.double)
        r,_ = cv2.Rodrigues(rotations[i])
        e[0:3] = r.reshape(3,1)
        e[3:6] = motions[i].reshape(3,1)
        # print (e)
        extrinsics.append(e)

    # print (rotations)
    # print (motions)

    key_points_for_all_list = []
    for i in range(len(key_points_for_all)):
        key_point_ = np.empty([0, 3])
        for kp in key_points_for_all[i]:
            key_point_ = np.vstack((key_point_, np.array([[kp.pt[0], kp.pt[1], kp.size]])))
        key_points_for_all_list.append(key_point_)

    # print (structure)
    structure_list = []
    for i in range(structure.shape[0]):
        structure_list.append(structure[i])

    # print ("correspond_struct_idx:{}".format(correspond_struct_idx))




    ba = BundleAdjustment.BA(intrinsic, extrinsics, correspond_struct_idx, key_points_for_all_list, structure_list)
    ba.save(data_path+ "structure.yml", rotations, motions, colors)
    # print (motions)
    #bundle_adjustment  # !!
    #save_structure  # !!

if __name__ == '__main__':
    main()

