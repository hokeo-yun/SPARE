import os

ForenSynths = [
    dict(
        real_path=os.path.join('/home/108/u108009/dataset/ForenSynths/test', dataset_name),
        fake_path=os.path.join('/home/108/u108009/dataset/ForenSynths/test', dataset_name),
        data_mode='wang2020',
        key=dataset_name,
        is_resize=False,
    ) for dataset_name in [
            "progan",
            "stylegan",
            "stylegan2",
            "biggan",
            "cyclegan",
            "stargan",
            "gaugan",
            "deepfake",
            "san",
            "diffusion_datasets/ldm_100",
            "diffusion_datasets/ldm_200",
            "diffusion_datasets/ldm_200_cfg",
            "diffusion_datasets/guided",
            "diffusion_datasets/glide_50_27",
            "diffusion_datasets/glide_100_10",
            "diffusion_datasets/glide_100_27",
            "diffusion_datasets/dalle",
        ]
]

GenImage = [
    dict(
        real_path=os.path.join('/home/108/u108009/dataset/GenImage/test', dataset_name),
        fake_path=os.path.join('/home/108/u108009/dataset/GenImage/test', dataset_name),
        data_mode='wang2020',
        key=dataset_name,
        is_resize=False,
    ) for dataset_name in [
        'ADM', 
        'Midjourney', 
        'stable_diffusion_v_1_4', 
        'glide', 
        'stable_diffusion_v_1_5', 
        'VQDM', 
        'wukong',
        'BigGAN'
        ]
]

DRCT = [
    dict(
        real_path=os.path.join('/home/108/u108009/dataset/DRCT-2M/test', dataset_name),
        fake_path=os.path.join('/home/108/u108009/dataset/DRCT-2M/test', dataset_name),
        data_mode='wang2020',
        key=dataset_name,
        is_resize=True,
    ) for dataset_name in os.listdir("/home/108/u108009/dataset/DRCT-2M/test")
]
