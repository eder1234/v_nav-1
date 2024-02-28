import open3d as o3d
import numpy as np
import os
import random
from scipy.spatial.transform import Rotation as R

class Registration:
    def __init__(self, config, pc1=None, pc2=None):
            self.config = config
            self.pc1 = pc1
            self.pc2 = pc2
            self.initial_pose = np.eye(4)
            self.aligned_pose = np.eye(4)
            self.max_iteration = config['registration']['max_iteration']
            self.distance_threshold = config['registration']['distance_threshold']
            self.global_fitness = 0
            self.icp_result = None
            self.voxel_size = config['registration']['voxel_size'] # this line could be deprecated
            self.fit_threshold = config['navigation']['fit_threshold']
            self.forward_threshold = config['navigation']['forward_threshold']
            self.lateral_threshold = config['navigation']['lateral_threshold']
            self.yaw_threshold = config['navigation']['yaw_threshold']

    def print_info_in_blue(self, message):
        blue = "\033[94m"  # ANSI escape code for blue
        reset = "\033[0m"  # Reset to default terminal color
        print(f"{blue}Info: {message}{reset}")

    def estimate_knn_radius(self, point_cloud, k=3):
        # Convert Open3D PointCloud to numpy array
        points = np.asarray(point_cloud.points)
        
        # Build a KDTree for efficient nearest neighbor search
        kdtree = o3d.geometry.KDTreeFlann(point_cloud)
        
        distances = []
        for i in range(len(points)):
            # For each point, find its k nearest neighbors (including itself, hence k+1)
            _, idx, _ = kdtree.search_knn_vector_3d(point_cloud.points[i], k + 1)
            
            # Calculate the distances to the k nearest neighbors (excluding the point itself)
            knn_distances = np.sqrt(np.sum((points[i] - points[idx[1:], :]) ** 2, axis=1))
            
            # Add the average distance to the list
            distances.append(np.mean(knn_distances))
        
        # Compute the overall average distance
        average_distance = np.mean(distances)
        
        return average_distance

    def compute_max_correspondence_distance(self, pc1, pc2):
        # Compute the axis-aligned bounding boxes for pc1 and pc2
        aabb_pc1 = pc1.get_axis_aligned_bounding_box()
        aabb_pc2 = pc2.get_axis_aligned_bounding_box()
        
        # Calculate the diagonal lengths of the bounding boxes
        diagonal_pc1 = np.linalg.norm(aabb_pc1.get_extent())
        diagonal_pc2 = np.linalg.norm(aabb_pc2.get_extent())
        
        # Use the average of the two diagonals as the basis for max_correspondence_distance
        max_correspondence_distance = np.mean([diagonal_pc1, diagonal_pc2])
        
        # Adjust the scale factor as necessary, depending on the expected alignment quality and point cloud density
        scale_factor = 0.1  # Example scale factor
        max_correspondence_distance *= scale_factor
        
        return max_correspondence_distance
    
    def estimate_optimal_voxel_size(self, point_cloud, k=1):
        """
        Estimate the optimal voxel size for downsampling.
        
        Args:
            point_cloud (open3d.geometry.PointCloud): The input point cloud.
            k (int): The number of nearest neighbors to consider, typically 1.

        Returns:
            float: The estimated optimal voxel size.
        """
        # Convert Open3D PointCloud to numpy array
        points = np.asarray(point_cloud.points)
        
        # Build a KDTree for efficient nearest neighbor search
        kdtree = o3d.geometry.KDTreeFlann(point_cloud)
        
        distances = []
        for i in range(len(points)):
            # For each point, find its nearest neighbor
            _, idx, dist = kdtree.search_knn_vector_3d(point_cloud.points[i], k + 1)
            
            # Exclude the point itself and take the distance to the nearest neighbor
            distances.append(np.sqrt(dist[1]))  # dist[0] is the distance to itself, which is 0

        # Use the median of these distances as the optimal voxel size
        optimal_voxel_size = np.median(distances)

        return optimal_voxel_size

    def estimate_initial_alignment(self): # To improve
        """
        Estimate initial alignment using FPFH features and RANSAC.

        Args:
        source (open3d.geometry.PointCloud): Source point cloud.
        target (open3d.geometry.PointCloud): Target point cloud.

        Returns:
        numpy.ndarray: The estimated transformation matrix.
        """
        if self.config['registration']['alignment'] == 'FPFH':
            result = self.estimate_initial_alignment_ransac() # To be reimplemented
            return result
        if self.config['registration']['alignment'] == 'RSCS':
            result = self.estimate_initial_alignment_from_superpoints()


    def pc_registration_generalized(self):

        icp_result = o3d.pipelines.registration.registration_generalized_icp(
            self.pc1, self.pc2, self.distance_threshold, self.initial_pose,
            o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=self.max_iteration,  # Max number of iterations
                #relative_fitness=1e-6,  
                #relative_rmse=1e-6  
            )
        )
        return icp_result
    
    def pc_registration_point_to_point(self):

        icp_result = o3d.pipelines.registration.registration_icp(
            self.pc1, self.pc2, self.distance_threshold, self.initial_pose,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=self.max_iteration)
        )
        return icp_result
    
    def pc_registration_point_to_plane(self):

        # Compute normals for the target point cloud
        target_with_normals = self.compute_normals(self.pc2)
        icp_result = o3d.pipelines.registration.registration_icp(
            self.pc1, target_with_normals, self.distance_threshold, self.initial_pose,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=self.max_iteration)
        )
        return icp_result
    
    def compute_normals(self, point_cloud, search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)): # search_param could be modified and its params could be tuned

        point_cloud.estimate_normals(search_param=search_param)
        point_cloud.orient_normals_consistent_tangent_plane(50)  # Optional, for better normal orientation
        return point_cloud
    
    def pc_registration_colored(self):

        max_correspondence_distance = self.compute_max_correspondence_distance(self.pc1, self.pc2)

        # Downsample the point clouds
        source_down = self.pc1.voxel_down_sample(self.voxel_size)
        target_down = self.pc2.voxel_down_sample(self.voxel_size)

        # Estimate normals
        source_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size*5, max_nn=50))
        target_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size*5, max_nn=50))

        # Perform Colored-ICP
        result = o3d.pipelines.registration.registration_colored_icp(
            source_down, target_down, max_correspondence_distance=max_correspondence_distance, init=self.initial_pose,
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=self.max_iteration)
        )

        return result
    
    def estimate_optimal_voxel_size(self, point_cloud, k=1):

        # Convert Open3D PointCloud to numpy array
        points = np.asarray(point_cloud.points)
        
        # Build a KDTree for efficient nearest neighbor search
        kdtree = o3d.geometry.KDTreeFlann(point_cloud)
        
        distances = []
        for i in range(len(points)):
            # For each point, find its nearest neighbor
            _, idx, dist = kdtree.search_knn_vector_3d(point_cloud.points[i], k + 1)
            
            # Exclude the point itself and take the distance to the nearest neighbor
            distances.append(np.sqrt(dist[1]))  # dist[0] is the distance to itself, which is 0

        # Use the median of these distances as the optimal voxel size
        optimal_voxel_size = np.median(distances)

        return optimal_voxel_size

    def pc_registration(self, pc1, pc2):
        self.pc1 = pc1
        self.pc2 = pc2

        optimal_voxel_size1 = self.estimate_optimal_voxel_size(pc1)
        optimal_voxel_size2 = self.estimate_optimal_voxel_size(pc2)

        self.voxel_size = np.mean([optimal_voxel_size1, optimal_voxel_size2])

        if self.config['registration']['alignment'] == 'FPFH':
            global_result = self.estimate_initial_alignment_ransac()
            self.initial_pose = global_result.transformation

        if self.config['registration']['alignment'] == 'RSCS':
            self.initial_pose = self.estimate_initial_alignment_from_superpoints()

        if self.config['registration']['alignment'] == 'SVD':
            self.estimate_initial_alignment_svd()

        if self.config['registration']['icp'] == 'generalized': 
            self.icp_result = self.pc_registration_generalized()

        if self.config['registration']['icp'] == 'point_to_point': 
            self.icp_result = self.pc_registration_point_to_point()

        if self.config['registration']['icp'] == 'point_to_plane': 
            self.icp_result = self.pc_registration_point_to_plane()

        if self.config['registration']['icp'] == 'colored': 
            self.icp_result = self.pc_registration_colored()

        if self.config['registration']['visualization']:
            self.visualize_registration_result()

        return self.icp_result
    
    def estimate_initial_alignment_svd(self):
        # Ensure the point clouds are converted to numpy arrays
        A = np.asarray(self.pc1.points)
        B = np.asarray(self.pc2.points)

        # Calculate centroids and centered arrays
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)
        AA = A - centroid_A
        BB = B - centroid_B

        # Compute the matrix H
        H = np.dot(AA.T, BB)
        U, S, Vt = np.linalg.svd(H)

        # Compute rotation
        R = np.dot(Vt.T, U.T)
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = np.dot(Vt.T, U.T)

        # Compute translation
        t = centroid_B - np.dot(R, centroid_A)

        # Update the initial_pose of the class
        self.initial_pose = np.eye(4)
        self.initial_pose[:3, :3] = R
        self.initial_pose[:3, 3] = t

        # Calculate the error (RMSE) and update global_fitness
        self.global_fitness = self.calculate_rmse(A, B, R, t)

    def calculate_rmse(self, A, B, R_est, t_est):
        """
        Calculate the RMSE between two point clouds after aligning A with the estimated rotation and translation.

        Parameters:
        - A: numpy.ndarray, the original point cloud (Nx3)
        - B: numpy.ndarray, the target point cloud to align with (Nx3)
        - R_est: numpy.ndarray, the estimated rotation matrix (3x3)
        - t_est: numpy.ndarray, the estimated translation vector (3,)

        Returns:
        - rmse: float, the root mean square error between the aligned A and B
        """
        # Apply the estimated rotation and translation to point cloud A
        A_aligned = np.dot(A, R_est.T) + t_est
        # Calculate the differences between the aligned A and B
        diffs = A_aligned - B
        # Compute the squared differences
        sq_diffs = np.square(diffs)
        # Calculate the mean squared error
        mse = np.mean(sq_diffs)
        # Calculate the root mean square error
        rmse = np.sqrt(mse)

        return rmse

    def estimate_initial_alignment_ransac(self):
        print('RANSAC initial alignment')
        # Estimate normals for both point clouds
        self.pc1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 2, max_nn=30))
        self.pc2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 2, max_nn=30))

        # Downsample point clouds to speed up computation
        pc1_down = self.pc1.voxel_down_sample(voxel_size=self.voxel_size)
        pc2_down = self.pc2.voxel_down_sample(voxel_size=self.voxel_size)

        # Compute FPFH features for both downsampled point clouds
        radius_feature = self.voxel_size * 5
        fpfh_pc1 = o3d.pipelines.registration.compute_fpfh_feature(pc1_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        fpfh_pc2 = o3d.pipelines.registration.compute_fpfh_feature(pc2_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

        # Set RANSAC registration parameters
        distance_threshold = self.voxel_size * 1.5
        ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            pc1_down, pc2_down, fpfh_pc1, fpfh_pc2, True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4,
            [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
            o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
        )

        return ransac_result

    def save_initial_point_clouds(self, count_steps):
        if self.config['logs']['initial_pc']:
            # Save the initial point clouds
            pc_root = self.config['logs']['merged_pc'] + 'initial_pc'
            os.makedirs(pc_root, exist_ok=True)
            pc_path = pc_root + f"/initial_pc_{count_steps:04d}.ply" 

            # Define red and blue colors
            red = np.array([1, 0, 0])  # RGB for red
            blue = np.array([0, 0, 1])  # RGB for blue

            # Assign red color to all points in pc1
            colors_pc1 = np.tile(red, (len(self.pc1.points), 1))
            self.pc1.colors = o3d.utility.Vector3dVector(colors_pc1)

            # Assign blue color to all points in pc2
            colors_pc2 = np.tile(blue, (len(self.pc2.points), 1))
            self.pc2.colors = o3d.utility.Vector3dVector(colors_pc2)

            # Merge the recolored point clouds
            merged_pc = self.pc1 + self.pc2

            o3d.io.write_point_cloud(pc_path, merged_pc)

    def visualize_registration_result(self):
        self.pc1.transform(self.icp_result.transformation)
        o3d.visualization.draw_geometries([self.pc1, self.pc2])
    
    def save_colored_registration_result(self, count_steps):
        if self.config['logs']['merged_pc']:
            # Define red and blue colors
            red = np.array([1, 0, 0])  # RGB for red
            blue = np.array([0, 0, 1])  # RGB for blue

            # Transform pc1 according to the ICP result
            self.pc1.transform(self.icp_result.transformation)

            # Assign red color to all points in pc1
            colors_pc1 = np.tile(red, (len(self.pc1.points), 1))
            self.pc1.colors = o3d.utility.Vector3dVector(colors_pc1)

            # Assign blue color to all points in pc2
            colors_pc2 = np.tile(blue, (len(self.pc2.points), 1))
            self.pc2.colors = o3d.utility.Vector3dVector(colors_pc2)

            # Merge the recolored point clouds
            merged_pc = self.pc1 + self.pc2

        # Save the merged and recolored point cloud
        self.save_merged_and_recolored_pc(merged_pc, count_steps)

    def save_merged_and_recolored_pc(self, point_cloud, count_steps):
        # Define the directory and filename for saving the point cloud
        pc_dir = self.config['paths']['LOGS_DIR'] + 'registration_colored_point_clouds'
        os.makedirs(pc_dir, exist_ok=True)
        pc_path = pc_dir + f"/registration_colored_pc_{count_steps:04d}.ply"  # Random ID for uniqueness
        
        # Save the point cloud as a .ply file
        o3d.io.write_point_cloud(pc_path, point_cloud)
        print(f"Registration colored point cloud saved to: {pc_path}")

    ### Experimental from this point #BeCareful
    
    def apply_RSCS(self, point_cloud, coverageLim=0.95, coverSphereRad=1):
        # Convert Open3D point cloud to NumPy array
        pntCloud_np = np.asarray(point_cloud.points)
        
        # Apply the RSCS method (assuming an adapted RSCS function named `createRandomSphereCoverSet`)
        superPntList, numSP = self.createRandomSphereCoverSet(pntCloud_np, coverageLim, coverSphereRad)
        
        # The `superPntList` contains the points with their corresponding 'SuperPoint' labels
        # This information could be used further for matching and registration purposes
        
        return superPntList, numSP

    def match_super_points(self, superPntList1, superPntList2):
        # Calculate centroids of 'SuperPoints' in both lists
        centroids1 = self.calculate_super_point_centroids(superPntList1)
        centroids2 = self.calculate_super_point_centroids(superPntList2)

        # Perform matching based on proximity (e.g., nearest neighbor)
        # This is a simplistic approach; more sophisticated methods might consider shape, size, etc.
        matched_pairs = []
        for sp_id1, centroid1 in centroids1.items():
            closest_sp_id2 = min(centroids2.keys(), key=lambda sp_id2: np.linalg.norm(centroid1 - centroids2[sp_id2]))
            matched_pairs.append((sp_id1, closest_sp_id2))

        return matched_pairs
    
    def calculate_super_point_centroids(self, superPntList):
        # Initialize a dictionary to hold 'SuperPoint' IDs and their corresponding points
        super_points = {}
        for point, label_list in superPntList:
            for label in label_list:
                if label in super_points:
                    super_points[label].append(point)
                else:
                    super_points[label] = [point]
        
        # Calculate centroids
        centroids = {sp_id: np.mean(points, axis=0) for sp_id, points in super_points.items()}
        
        return centroids

    def estimate_initial_alignment_from_superpoints(self):
        print("Estimating initial alignment from SuperPoints...")
        # Apply RSCS to both point clouds to generate 'SuperPoints'
        superPntList1, _ = self.apply_RSCS(self.pc1, 
                                        coverageLim=self.config['rscs']['coverageLim'], 
                                        coverSphereRad=self.config['rscs']['coverSphereRad'])
        superPntList2, _ = self.apply_RSCS(self.pc2, 
                                        coverageLim=self.config['rscs']['coverageLim'], 
                                        coverSphereRad=self.config['rscs']['coverSphereRad'])

        # Match 'SuperPoints' between the two point clouds
        matched_pairs = self.match_super_points(superPntList1, superPntList2)

        # Calculate centroids for 'SuperPoints' for use in initial alignment
        centroids1 = self.calculate_super_point_centroids(superPntList1)
        centroids2 = self.calculate_super_point_centroids(superPntList2)

        # Extract matching centroids
        source_points = np.array([centroids1[sp_id1] for sp_id1, _ in matched_pairs])
        target_points = np.array([centroids2[sp_id2] for _, sp_id2 in matched_pairs])

        # Ensure there are enough matches to compute a reliable transformation
        if len(source_points) < 3 or len(target_points) < 3:
            print("Not enough matches to estimate a reliable initial alignment.")
            return np.eye(4)  # Return identity matrix as a fallback

        # Estimate the transformation matrix using Singular Value Decomposition (SVD)
        H = source_points.T @ target_points
        U, S, Vt = np.linalg.svd(H)
        R_matrix = Vt.T @ U.T  # Rotation matrix

        rotation = R.from_matrix(R_matrix)

        # Calculate the translation vector
        centroid_source = np.mean(source_points, axis=0)
        centroid_target = np.mean(target_points, axis=0)
        t = centroid_target - R_matrix @ centroid_source  # Translation vector

        # Construct the transformation matrix
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = R_matrix
        transformation_matrix[:3, 3] = t

        #self.print_info_in_blue("Initial alignment from RSCS estimated.")
        euler_angles = rotation.as_euler('xyz', degrees=True)

        #self.print_info_in_blue(f"Translation: {t}")
        #self.print_info_in_blue(f"Angles (xyz): {euler_angles}")

        suggested_action = self.suggested_action_from_global(t, euler_angles)
        self.print_info_in_blue(f"Suggested action: {suggested_action}")

        return transformation_matrix


    def refine_alignment_with_icp(self, initial_transformation):
        # Use the initial transformation as the starting point for ICP
        icp_result = o3d.pipelines.registration.registration_icp(
            self.pc1, self.pc2, self.distance_threshold, initial_transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=self.max_iteration)
        )
        return icp_result.transformation

    def pc_registration_with_superpoints(self):
        # Apply RSCS to both point clouds
        superPntList1, _ = self.apply_RSCS(self.pc1)
        superPntList2, _ = self.apply_RSCS(self.pc2)

        # Calculate centroids for 'SuperPoints'
        centroids1 = self.calculate_super_point_centroids(superPntList1)
        centroids2 = self.calculate_super_point_centroids(superPntList2)

        # Match 'SuperPoints'
        matched_pairs = self.match_super_points(superPntList1, superPntList2)

        # Estimate initial alignment from 'SuperPoints'
        initial_transformation = self.estimate_initial_alignment_from_superpoints(matched_pairs, centroids1, centroids2)

        # Refine alignment with ICP
        refined_transformation = self.refine_alignment_with_icp(initial_transformation)

        # Update the aligned pose
        self.aligned_pose = refined_transformation

        # Optionally visualize or return the result
        # self.visualize_registration_result()  # Uncomment if you want to visualize the result
        return initial_transformation

    def createRandomSphereCoverSet(self, pntCloud_np, coverageLim=0.95, coverSphereRad=1):
        # Assuming `pntCloud_np` is already a NumPy array of points, so no need to convert
        pntCloudFull = self.getLabeledFormat(pntCloud_np)
        spLabel = 0
        while self.getCoveragePercent(pntCloudFull) < coverageLim:
            centerPnt = self.chooseRandNonCoveredPnt(pntCloudFull)
            pntCloudFull = self.labelPointsInRad(centerPnt, coverSphereRad, pntCloudFull, spLabel)
            spLabel += 1
        return pntCloudFull, spLabel

    def getLabeledFormat(self, pntCloud):
        # Initialize points with an empty list for labels
        return [(point, []) for point in pntCloud]

    def getCoveragePercent(self, pntCloudFull):
        # Calculate the percentage of points that have been covered (labeled)
        coveredPoints = sum(1 for _, labels in pntCloudFull if labels)
        totalPoints = len(pntCloudFull)
        return coveredPoints / totalPoints if totalPoints else 0

    def chooseRandNonCoveredPnt(self, pntCloudFull):
        # Select a random point that hasn't been covered yet
        nonCoveredPoints = [point for point, labels in pntCloudFull if not labels]
        return random.choice(nonCoveredPoints) if nonCoveredPoints else None

    def labelPointsInRad(self, centerPnt, coverSphereRad, pntCloud, labelNum):
        # Label points within a specified radius of a center point
        for i, (point, labels) in enumerate(pntCloud):
            if np.linalg.norm(np.array(point) - np.array(centerPnt)) <= coverSphereRad:
                pntCloud[i][1].append(labelNum)
        return pntCloud

    def suggested_action_from_global(self, translation, euler_angles):
        # Check translation for forward/backward movement
        if translation[0] < -self.forward_threshold:
            action_forward = 'Move Forward'
        else:
            action_forward = 'Stop'  # If the bot is close enough to the target

        # Check lateral translation and yaw angle for turning
        if translation[1] < -self.lateral_threshold or euler_angles[2] < -self.yaw_threshold:
            action_turn = 'Turn Right'
        elif translation[1] > self.lateral_threshold or euler_angles[2] > self.yaw_threshold:
            action_turn = 'Turn Left'
        else:
            action_turn = None  # No turn is needed if within thresholds

        # Combine actions: prioritize turning over moving forward
        if action_turn:
            return action_turn
        else:
            return action_forward