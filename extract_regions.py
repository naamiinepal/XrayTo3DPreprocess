from pathlib import Path
import argparse

from argparse import ArgumentParser

from preprocessing import load_centroids,read_image, ImagePixelType, extract_around_centroid, write_image

parser = ArgumentParser(description='extract region whose centroid are given')
parser.add_argument('--ct',required=True)
parser.add_argument('--seg',required=True)
parser.add_argument('--centroids',required=True)
parser.add_argument('-o',required=True)

args = parser.parse_args()

print(args)


ctd_list = load_centroids(args.centroids)
img = read_image(args.ct)
seg = read_image(args.seg)

print(ctd_list)
vb_id,*ctds = ctd_list[5]
print(vb_id,ctds)

roi = extract_around_centroid(img, (100,100,100), ctds,-1024)
write_image(roi,args.o)