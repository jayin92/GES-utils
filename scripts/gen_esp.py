import argparse
import json
import pymap3d
import numpy as np

from geopy.distance import geodesic
from geopy.point import Point

def get_coordinates(lat, lon, offset, sample):
    # lat lon is the bottom left corner
    # therefore, we need to calculate the top right corner

    bottom_right = geodesic(kilometers=offset).destination(Point(lat, lon), 90)
    top_right = geodesic(kilometers=offset).destination(bottom_right, 0)

    lat_0 = lat
    lon_0 = lon
    lat_1 = top_right.latitude
    lon_1 = top_right.longitude
    
    print(f"Offset: {offset} km")
    print(f"Bottom left: {lat_0}, {lon_0}")
    print(f"Top right: {lat_1}, {lon_1}")

    X, Y = np.meshgrid(np.linspace(lat_0, lat_1, sample), np.linspace(lon_0, lon_1, sample))
    coordinates = np.vstack([X.ravel(), Y.ravel()]).T
    
    print(coordinates.shape)

    return coordinates
    
    
    # Calculate the coordinates

def convert_to_relative(coordinates, altitude, tilt):
    # Define the MinValueRange and MaxValueRange for each dimension
    min_longitude, max_longitude = -180, 180
    min_latitude, max_latitude = -90, 90
    min_altitude, max_altitude = -500, 65117481
    min_tilt, max_tilt = 0, 180


    num_points = len(coordinates) * 4
    
    # Calculate the indices to extract equally distributed points

    # Initialize keyframes for longitude, latitude, and altitude
    keyframes_longitude = []
    keyframes_latitude = []
    keyframes_altitude = []
    keyframes_rotationX = [] # Pan
    keyframes_rotationY = [] # Tilt
    for i, coord in enumerate(coordinates):
        latitude, longitude = coord[0], coord[1]
        relative_longitude = (longitude - min_longitude) / (max_longitude - min_longitude)
        relative_latitude = (latitude - min_latitude) / (max_latitude - min_latitude)
        relative_altitude = (altitude - min_altitude) / (max_altitude - min_altitude)
        relative_tilt = (tilt - min_tilt) / (max_tilt - min_tilt)
        for j in range(4):
            idx = i * 4 + j
            # Create keyframe for longitude
            keyframe_longitude = {
                "time": idx / (num_points - 1),
                "value": relative_longitude
            }
            keyframes_longitude.append(keyframe_longitude)

            # Create keyframe for latitude
            keyframe_latitude = {
                "time": idx / (num_points - 1),
                "value": relative_latitude
            }
            keyframes_latitude.append(keyframe_latitude)

            # Create keyframe for altitude
            keyframe_altitude = {
                "time": idx / (num_points - 1),
                "value": relative_altitude
            }
            keyframes_altitude.append(keyframe_altitude)

            # Create keyframe for rotationX
            keyframe_rotationX = {
                "time": idx / (num_points - 1),
                "value": 0.25 * j
            }
            keyframes_rotationX.append(keyframe_rotationX)

            # Create keyframe for rotationY
            keyframe_rotationY = {
                "time": idx / (num_points - 1),
                "value": relative_tilt
            }
            keyframes_rotationY.append(keyframe_rotationY)



    return keyframes_longitude, keyframes_latitude, keyframes_altitude, keyframes_rotationX, keyframes_rotationY

def generate_esp_content(lat, lon, offset, keyframes_longitude, keyframes_latitude, keyframes_altitude, keyframes_rotationX, keyframes_rotationY):
    num_frames = len(keyframes_longitude)-1
    esp_content = {
        "modelVersion": 18,
        "settings": {
            "name": f"{lat}_{lon}_{offset}",
            "frameRate": 30,
            "dimensions": {
                "width": 1920,
                "height": 1080
            },
            "duration": num_frames,
            "timeFormat": "frames"
        },
        "scenes": [
            {
                "animationModel": {
                    "roving": False,
                    "logarithmic": False,
                    "groupedPosition": True
                },
                "duration": num_frames,
                "attributes": [
                    {
                        "type": "cameraGroup",
                        "inTimeline": True,
                        "attributes": [
                            {
                                "type": "cameraPositionGroup",
                                "inTimeline": True,
                                "attributes": [
                                    {
                                        "type": "position",
                                        "inTimeline": True,
                                        "attributes": [
                                            {
                                                "type": "longitude",
                                                "value": {
                                                    "relative": 0.8823162460219629
                                                },
                                                "keyframes": keyframes_longitude,
                                                "inTimeline": True
                                            },
                                            {
                                                "type": "latitude",
                                                "value": {
                                                    "relative": 0.6955960523121039
                                                },
                                                "keyframes": keyframes_latitude,
                                                "inTimeline": True
                                            },
                                            {
                                                "type": "altitude",
                                                "value": {
                                                    "maxValueRange": 65117481,
                                                    "minValueRange": -500,
                                                    "relative": 0.000035469247731679194,
                                                    "logarithmic": False
                                                },
                                                "keyframes": keyframes_altitude,
                                                "inTimeline": True
                                            }
                                        ]
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "type": "cameraRotationGroup",
                        "inTimeline": True,
                        "attributes": [
                            {
                                "type": "rotationX",
                                "value": {
                                    "maxValueRange": 360,
                                    "minValueRange": 0,
                                    "relative": 0
                                },
                                "keyframes": keyframes_rotationX,
                                "inTimeline": True
                            },
                            {
                                "type": "rotationY",
                                "value": {
                                    "relative": 0.4198690182100391
                                },
                                "keyframes": keyframes_rotationY,
                                "inTimeline": True
                            },
                            {
                                "type": "rotationZ",
                                "value": {
                                    "relative": 2.7751497430848027e-11
                                }
                            }
                        ]
                    }
                    # Other scene attributes...
                ],
                "cameraExport": {
                    "logarithmic": False,
                    "modelVersion": 2
                }
            }
        ],
        "playbackManager": {
            "range": {
                "start": 0,
                "end": num_frames
            }
        }
    }

    return esp_content

def write_to_esp_file(esp_content, output_path):
    with open(output_path, 'w') as output_file:
        json.dump(esp_content, output_file, indent=2)

    print(f"Result has been written to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate TrackPoints.esp file from XML coordinates.')
    parser.add_argument('--lat', type=float, help='Bottom left Latitude', required=True)
    parser.add_argument('--lon', type=float, help='Bottom right Longitude', required=True)
    parser.add_argument('--alt', type=float, help='Altitude in meter', required=True)
    parser.add_argument('--tilt', type=float, help='Tilt in degree (0 ~ 180)', required=True)
    parser.add_argument('--offset', type=float, help='Cover range in km', required=True)
    parser.add_argument('--sample', type=int, help='Number of samples in one dimesion', default=10)
    parser.add_argument('--output', type=str, help='Output file', required=True)
    args = parser.parse_args()

    coordinates = get_coordinates(args.lat, args.lon, args.offset, args.sample)
    keyframes_longitude, keyframes_latitude, keyframes_altitude, keyframes_rotationX, keyframes_rotationY = convert_to_relative(coordinates, args.alt, args.tilt)
    esp_content = generate_esp_content(args.lat, args.lon, args.offset, keyframes_longitude, keyframes_latitude, keyframes_altitude, keyframes_rotationX, keyframes_rotationY)
    write_to_esp_file(esp_content, args.output)
    