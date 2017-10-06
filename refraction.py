import trimesh
import numpy as np
import math

def main():
    transform = np.eye(4)
    transform[0,3]=1
    mesh = trimesh.primitives.Box(extents=np.array([1,2,3]), transform=transform)

    ray_origin = np.array([[0,0,0]])
    ray_direction = normalize(np.array([[1,0.1,0]]))

    ret = mesh.ray.intersects_location(ray_origin, ray_direction)
    triangles, rays, locations = mesh.ray.intersects_id(ray_origin, ray_direction, return_locations=True)

    distances = []
    for i, location in enumerate(locations):
        distance = np.linalg.norm(location - ray_origin)
        distances.append(distance)

    intersections_sorted = sorted(zip(distances, triangles, locations), key=lambda x: x[0], reverse=False)

    triangle_nearest = intersections_sorted[0][1]
    location_nearest = intersections_sorted[0][2]
    normal_nearest = mesh.face_normals[triangle_nearest]

    print("Location " +str(location_nearest) + " normal direction " + str(normal_nearest))

    axis_rotation = np.cross(ray_direction, normal_nearest)
    axis_rotation_norm = axis_rotation/np.linalg.norm(axis_rotation)
    print("Rotation axis: " + str(axis_rotation_norm))


    angle_incident = math.acos(np.dot(ray_direction, normal_nearest) / (np.linalg.norm(ray_direction) * np.linalg.norm(normal_nearest)))
    angle = calculate_refraction_angle(angle_incident, 1.0, 1.2)
    print("Angle 1: " + str(angle_incident*180/np.pi) + " Angle 2: " + str(angle*180/np.pi))

    # rotation_euler = trimesh.transformations.euler_from_quaternion(trimesh.transformations.quaternion_about_axis(np.pi/4, axis_rotation_norm.T))
    rotation_mat = trimesh.transformations.rotation_matrix(angle, axis_rotation_norm.T)
    print(rotation_mat)
    print("Original direction: " + str(ray_direction) + " Rotated direction: " + str(ray_direction * rotation_mat[0:3,0:3]))
    print(np.linalg.norm(ray_direction * rotation_mat[0:3,0:3]))

def calculate_refraction_angle(angle_in, index_outer, index_inner):
    angle_out = math.asin((index_outer/index_inner)*math.sin(angle_in))
    return angle_out

def normalize(input):
    return input/np.linalg.norm(input)

if __name__ == '__main__':
    main()
