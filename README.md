# GES-utils
NeRF Training Utils for Google Earth Studio


## Scripts

- `ges2nerf.py`: Transform 3D Tracking json to LandMark (Grid-NeRF) compatitable format (transform.json)
- `prep_ges_datasets.py`: Combine two videos to remove watermark, also support lowering the resolution
- `gen_esp.py`: Generate .esp file (GES Project File) for given latitude, longitude, altitude, offset and camera tilt angle.

## Roadmap

1. Directly inherit NeRFStudio to create a GES Dataloader
2. Web UI for generating ESP file more starightforward
