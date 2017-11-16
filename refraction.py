import trimesh
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def main():
    phantom_transform = np.eye(4)
    phantom_transform[1,3]=0.12
    print(phantom_transform)
    # mesh_phantom = trimesh.primitives.Box(extents=np.array([0.12675,0.0579,0.0579]), transform=transform)
    phantom_dims = np.array([0.12675,0.0579,0.0579])
    # mesh_phantom.show()

    camera_a_origin = np.array([0,0.12,0.12])
    camera_b_origin = np.array([0,0,0])

    point_observed = np.array([0.05,0.12 + 0.025,0-0.025])

    modeler = RefractionModeler(camera_a_origin, camera_b_origin, phantom_dims, phantom_transform, 1.2, 1.0)

    ret, real_point = modeler.solve_real_point_from_refracted(point_observed)
    print("Observed Point", point_observed, "Real Point", real_point)
    # modeler.make_plot()

class RefractionModeler(object):
    def __init__(self, camera_a_origin, camera_b_origin, phantom_mesh_dims, phantom_transform, refractive_index_phantom, refractive_index_ambient):
        self.camera_a_origin = camera_a_origin
        self.camera_b_origin = camera_b_origin
        self.mesh_phantom = trimesh.primitives.Box(extents=phantom_mesh_dims, transform=phantom_transform)
        self.refractive_index_phantom = refractive_index_phantom
        self.refractive_index_ambient = refractive_index_ambient
        # self.fig = plt.figure()
        # self.ax = self.fig.add_subplot(111, projection='3d')

    def solve_real_point_from_refracted(self, point_observed):
        # print("Origin A", self.camera_a_origin)
        # print("Origin B", self.camera_b_origin)
        self.point_observed = point_observed
        # print("Point observed", point_observed)

        camera_a_direction = self._normalize(point_observed - self.camera_a_origin)
        camera_b_direction = self._normalize(point_observed - self.camera_b_origin)

        # print("Dir A",camera_a_direction)
        # print("Dir B",camera_b_direction)

        success_a, location_nearest_a, normal_nearest_a = self._get_closest_intersection(self.camera_a_origin, camera_a_direction)
        success_b, location_nearest_b, normal_nearest_b = self._get_closest_intersection(self.camera_b_origin, camera_b_direction)
        if not success_a or not success_b:
            success = False
            self.real_point = None
        else:
            # print("Inter A", location_nearest_a)
            # print("Inter B", location_nearest_b)

            # print("Norm A", normal_nearest_a)
            # print("Norm B", normal_nearest_b)

            camera_a_direction_refracted = self._get_refracted_direction(camera_a_direction, normal_nearest_a)
            camera_b_direction_refracted = self._get_refracted_direction(camera_b_direction, normal_nearest_b)

            # print("New Dir A", camera_a_direction_refracted)
            # print("New Dir B", camera_b_direction_refracted)

            self.real_point, point_a, point_b, delta = self._rays_closest_point(location_nearest_a, camera_a_direction_refracted, location_nearest_b, camera_b_direction_refracted)
            self.lines_a = np.concatenate(([self.camera_a_origin], [location_nearest_a], [point_a]), axis=0)
            self.lines_b = np.concatenate(([self.camera_b_origin], [location_nearest_b], [point_b]), axis=0)

            # print("Refraction error", np.linalg.norm(self.real_point - self.point_observed))
            success = True
        return success, self.real_point

    def make_plot(self):
        self.ax.clear()
        self.ax.plot3D(self.lines_a[:, 0], self.lines_a[:, 1], self.lines_a[:, 2])
        self.ax.plot3D(self.lines_b[:, 0], self.lines_b[:, 1], self.lines_b[:, 2])
        self.ax.scatter(self.camera_a_origin[0], self.camera_a_origin[1], self.camera_a_origin[2])
        self.ax.scatter(self.camera_b_origin[0], self.camera_b_origin[1], self.camera_b_origin[2])
        self.ax.scatter(self.real_point[0], self.real_point[1], self.real_point[2], 'r')
        self.ax.scatter(self.point_observed[0], self.point_observed[1], self.point_observed[2], 'g')
        plt.axis('equal')
        plt.xlabel("X")
        plt.ylabel("Y")
        # plt.draw()  # redraw the canvas
        # self.fig.canvas.flush_events()

    def _get_closest_intersection(self, ray_origin, ray_direction):
        triangles, rays, locations = self.mesh_phantom.ray.intersects_id(np.reshape(ray_origin,(1,3)), np.reshape(ray_direction,(1,3)), return_locations=True)
        distances = []
        for i, location in enumerate(locations):
            distance = np.linalg.norm(location - ray_origin)
            distances.append(distance)
        if len(distances) > 0:
            intersection_found = True
            intersections_sorted = sorted(zip(distances, triangles, locations), key=lambda x: x[0], reverse=False)
            # print("Intersections sorted", intersections_sorted)
            triangle_nearest = intersections_sorted[0][1]
            location_nearest = intersections_sorted[0][2]
            # print("Loc Nearest", location_nearest)
            normal_nearest = self.mesh_phantom.face_normals[triangle_nearest]
        else:
            intersection_found = False
            location_nearest = None
            normal_nearest = None
        return intersection_found, location_nearest, normal_nearest

    def _get_refracted_direction(self, ray_direction, normal_nearest):
        axis_rotation = np.cross(ray_direction, normal_nearest)
        axis_rotation_norm = axis_rotation / np.linalg.norm(axis_rotation)
        angle_incident = math.acos(
            np.dot(ray_direction, -normal_nearest) / (np.linalg.norm(ray_direction) * np.linalg.norm(-normal_nearest)))
        angle = self._calculate_refraction_angle(angle_incident, self.refractive_index_ambient, self.refractive_index_phantom)
        angle_diff = angle - angle_incident
        # print(angle_diff)
        if abs(angle_diff) > 0.0:
            rotation_mat = trimesh.transformations.rotation_matrix(angle_diff, axis_rotation_norm.T)
        else:
            rotation_mat = np.eye(3)
        ray_direction_refracted = np.dot(rotation_mat[0:3, 0:3], ray_direction.T)
        return ray_direction_refracted

    def _rays_closest_point(self, ray1_origin, ray1_direction, ray2_origin, ray2_direction):
        # http://morroworks.com/Content/Docs/Rays%20closest%20point.pdf
        c = ray2_origin - ray1_origin
        D = ray1_origin + ray1_direction * (-np.dot(ray1_direction, ray2_direction) * np.dot(ray2_direction, c)
                                             + np.dot(ray1_direction, c) * np.dot(ray2_direction, ray2_direction)) / \
                          (np.dot(ray1_direction, ray1_direction) * np.dot(ray2_direction, ray2_direction)
                                            - np.dot(ray1_direction, ray2_direction) * np.dot(ray1_direction, ray2_direction))
        E = ray2_origin + ray2_direction * (np.dot(ray1_direction, ray2_direction) * np.dot(ray1_direction, c) - np.dot(
                                                ray2_direction, c) * np.dot(ray1_direction, ray1_direction)) / \
                          (np.dot(ray1_direction, ray1_direction) * np.dot(ray2_direction,ray2_direction)
                           - np.dot(ray1_direction, ray2_direction) * np.dot(ray1_direction, ray2_direction))
        return (D + E) * 0.5, D, E, np.linalg.norm(E-D)

    def _normalize(self, input):
        return np.array(input / np.linalg.norm(input))

    def _calculate_refraction_angle(self, angle_in, index_outer, index_inner):
        angle_out = math.asin((index_outer / index_inner) * math.sin(angle_in))
        return angle_out


if __name__ == '__main__':
    main()
