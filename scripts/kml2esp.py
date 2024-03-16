# COPYRIGHT Pat Wilson
# Download from https://groups.google.com/g/earthstudio-discuss/c/N874xyR3g6M/m/nI5rezzpAAAJ

# This script converts a KML file to a GES file. The KML file should contain a single LineString with coordinates. The GES file will contain a camera animation that follows the path defined by the coordinates.

import xml.etree.ElementTree as ET
import json
import argparse

def convert_to_relative(coordinates, num_points, altitude_offset):
    # Define the MinValueRange and MaxValueRange for each dimension
    min_longitude, max_longitude = -180, 180
    min_latitude, max_latitude = -90, 90
    min_altitude, max_altitude = -500, 65117481

    total_points = len(coordinates)
    
    # Calculate the indices to extract equally distributed points
    indices = [int(i * (total_points - 1) / (num_points - 1)) for i in range(num_points)]

    # Initialize keyframes for longitude, latitude, and altitude
    keyframes_longitude = []
    keyframes_latitude = []
    keyframes_altitude = []

    for i, idx in enumerate(indices):
        coord = coordinates[idx]
        longitude, latitude, altitude = map(float, coord.split(','))

        # Add altitude_offset to the altitude
        altitude += altitude_offset

        # Convert to relative values
        relative_longitude = (longitude - min_longitude) / (max_longitude - min_longitude)
        relative_latitude = (latitude - min_latitude) / (max_latitude - min_latitude)
        relative_altitude = (altitude - min_altitude) / (max_altitude - min_altitude)

        # Create keyframe for longitude
        keyframe_longitude = {
            "time": i / (num_points - 1),
            "value": relative_longitude
        }
        keyframes_longitude.append(keyframe_longitude)

        # Create keyframe for latitude
        keyframe_latitude = {
            "time": i / (num_points - 1),
            "value": relative_latitude
        }
        keyframes_latitude.append(keyframe_latitude)

        # Create keyframe for altitude
        keyframe_altitude = {
            "time": i / (num_points - 1),
            "value": relative_altitude
        }
        keyframes_altitude.append(keyframe_altitude)

    return keyframes_longitude, keyframes_latitude, keyframes_altitude

def read_coordinates_from_xml(file_path):
    print(file_path)
    tree = ET.parse(file_path)
    root = tree.getroot()

    # print tag
    print(root.attrib)


    coordinates_str = root.text.strip()
    coordinates_list = coordinates_str.split()

    return coordinates_list

def generate_esp_content(keyframes_longitude, keyframes_latitude, keyframes_altitude):
    esp_content = {
        "modelVersion": 18,
        "settings": {
            "name": "Train",
            "frameRate": 30,
            "dimensions": {
                "width": 1920,
                "height": 1080
            },
            "duration": 450,
            "timeFormat": "frames"
        },
        "scenes": [
            {
                "animationModel": {
                    "roving": False,
                    "logarithmic": False,
                    "groupedPosition": True
                },
                "duration": 450,
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
                "end": 450
            }
        }
    }

    return esp_content

def write_to_esp_file(esp_content):
    output_file_path = 'TrackPoints.esp'
    with open(output_file_path, 'w') as output_file:
        json.dump(esp_content, output_file, indent=2)

    print(f"Result has been written to {output_file_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate TrackPoints.esp file from XML coordinates.')
    parser.add_argument('--input_file', default='coordinates.xml', help='Path to the input XML file')
    parser.add_argument('--num_points', type=int, default=10, help='Number of points in the output JSON file')
    parser.add_argument('--altitude_offset', type=float, default=0.0, help='Altitude offset to be added to the altitudes')

    args = parser.parse_args()

    # Read coordinates from XML file
    coordinates_str = read_coordinates_from_xml(args.input_file)

    # Convert to keyframes for longitude, latitude, and altitude
    keyframes_longitude, keyframes_latitude, keyframes_altitude = convert_to_relative(
        coordinates_str, args.num_points, args.altitude_offset
    )

    # Generate ESP content
    esp_content = generate_esp_content(keyframes_longitude, keyframes_latitude, keyframes_altitude)

    # Write to ESP file
    write_to_esp_file(esp_content)

if __name__ == "__main__":
    main()