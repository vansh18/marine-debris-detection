from PIL import Image
import sys
import os
import numpy as np
from skimage import measure
from shapely.geometry import Polygon, MultiPolygon
import json
import random
import cv2
import zlib, base64
from tqdm import tqdm
import matplotlib.pyplot as plt
from shutil import copy2 as copy

# collect all mask image file names and randomly split into train and val sets

dataset_version = sys.argv[1]
is_crowd = 0
annotation_id = 1
image_id = 1 
bio_fail = list()

trash_unknown_akas = ['unkonwn_instance', 'cage', 'uknwon_instance', 'unknwon_instance', 'unknown', 'unkown_instance', 'etc', 'unknown_instances', 'unkonwn_instances', 'eel', 'hay', 'tire', 'hose', 'wire', 'glove', 'box', 'bucket', 'paper']
bio_etc_akas       = ['turtle', 'squid', 'lobster', 'unknown', 'jellyfish', 'b', 'stingray', 'shrimp', 'crawfish', 'octopus', 'shark']
trash_instance_number_list = ['clothing', 'pipe', 'bottle', 'bag', 'snack_wrapper', 'glove', 'tire', 'can', 'cup','container', 'unknown_instance', 'branch', 'wreakage', 'tarp', 'box', 'hose', 'rope', 'hay', 'net', 'paper', 'bucket', 'wire']
bio_instance_number_list = ['fish', 'crab', 'eel', 'shark', 'squid', 'starfish', 'unknown', 'shell', 'shrimp', 'stingray', 'jellyfish', 'lobster', 'crawfish', 'octopus']

if dataset_version != 'material' and dataset_version != 'instance':
    raise ValueError('Dataset version needs to be either material or instance')



def base64_2_mask(s):
    z = zlib.decompress(base64.b64decode(s))
    n = np.fromstring(z, np.uint8)
    mask = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)[:, :, 3].astype(np.float)
    return mask

def submask_2_mask(submask, origin, height, width):
    mask = np.zeros((height+2, width+2), np.float)
    ox = int(origin[0]+1)
    oy = int(origin[1]+1)
    hy, wx = submask.shape
    mask[oy:oy+hy, ox:ox+wx] = submask

    return mask

def create_mask_annotation(im_name, mask, image_id, category_id, annotation_id, is_crowd, show_vis):
    # Find contours (boundary lines) around each sub-mask
    contours = measure.find_contours(mask, 0.5, positive_orientation='low')

    #TODO Fix contour issues.
    segmentations = []
    polygons = []

    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col, row)

        # Make a polygon and simplify it
        if len(contour) < 3:
            continue

        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=False)

        if (poly.geom_type == 'MultiPolygon'):
            # if MultiPolygon, take the smallest convex Polygon containing all the points in the object
            poly = poly.convex_hull
          
        if (poly.geom_type == 'Polygon'): # Ignore if still not a Polygon (could be a line or point)
            polygons.append(poly)
            segmentation = np.array(poly.exterior.coords).ravel().tolist()
            if segmentation != []:
                segmentations.append(segmentation)

    if show_vis:
        fig, ax = plt.subplots()
        plt.imshow(mask, cmap='gray')
        print(im_name)
        for c in contours:
            ax.plot(c[:, 0], c[:, 1], color='red', linewidth=2)
        for p in polygons:
            if not p.is_empty:
                plt.plot(*p.exterior.xy, color='yellow', linewidth=2)
        plt.show()
        plt.close()
    
 
    # Combine the polygons to calculate the bounding box and area
    multi_poly = MultiPolygon(polygons)
    x, y, max_x, max_y = multi_poly.bounds
    width = max_x - x
    height = max_y - y
    bbox = (x, y, width, height)
    area = multi_poly.area

    annotation = {
        'segmentation': segmentations,
        'iscrowd': is_crowd,
        'image_id': image_id,
        'category_id': category_id,
        'id': annotation_id,
        'bbox': bbox,
        'area': area
    }

    return annotation

def create_image_entry(file_name, im_id, width, height):
  return {
          "file_name": file_name,
          "height": height,
          "width": width,
          "id": im_id
      }

def get_category_id_string(obj, ann_file):
    global dataset_version, bio_type

    id_string = None
    if obj['geometryType'] != 'bitmap':
        return None    #Skip if the annotation type isn't bitmap

    class_title = str(obj['classTitle'])

    if class_title == 'rov':
        id_string = 'rov'

    elif class_title == 'unknown':
        if dataset_version == 'material':
            id_string = 'trash_etc'
        else:
            id_string = 'trash_unknown_instance'

    elif class_title == 'bio':
        tags = obj['tags']
        type_found = False
        for tag in tags:
            if tag['name'] == 'plant':
                id_string = 'plant'
                type_found = True
            elif tag['name'] == 'animal':
                tag_value = str(tag['value'])
                if tag_value.isdigit():
                    tag_value = bio_instance_number_list[int(tag_value)+1]
                
                if tag_value in bio_etc_akas:
                    id_string = 'animal_etc'
                elif tag_value == 'satrfish':
                    id_string = 'animal_starfish'
                elif tag_value == 'shell':
                    id_string = 'animal_shells'
                elif tag_value == 'ee;':
                    id_string = 'animal_eel'
                elif tag_value == ' crab':
                    id_string = 'animal_crab'
                else:
                    id_string = 'animal_' + str(tag_value)

                type_found = True
                    
        if not type_found:
            bio_fail.append(f"{ann_file} : {obj['tags']}")
            return None
                    
    elif class_title == 'trash':
        tags = obj['tags']

        if dataset_version == 'material':
            material_found = False
            for tag in tags:
                if tag['name'] == 'material':
                    if tag['value'] != 'glass' and tag['value'] != 'glass':
                       id_string = 'trash_'+str(tag['value']) #Create cateogry_id key from trash_ + material value.
                    else:
                        id_string =  'trash_etc' 

                    material_found = True

            if not material_found:
                id_string = 'trash_etc'

        else:
            instance_found = False
            for tag in tags:
                if tag['name'] == 'instance':
                    tag_value = tag['value']
                    if tag_value.isdigit():
                        tag_value = trash_instance_number_list[int(tag_value)+1]
                                
                    if tag_value == 'wreakage':
                        id_string = 'trash_wreckage' #Create cateogry_id key from trash_ + material value.
                    elif tag_value in trash_unknown_akas:
                        id_string =  'trash_unknown_instance'
                    elif tag_value == 'trap':
                        id_string = 'trash_tarp'
                    elif tag_value == 'bags' :
                        id_string =  'trash_bag'
                    elif tag_value == 'cab' :
                        id_string = 'trash_can'
                    elif tag_value == 'network' :
                        id_string = 'trash_net'
                    elif tag_value == '8\\':
                        id_string = 'trash_can'
                    else:
                        id_string =  'trash_'+str(tag_value) #Create cateogry_id key from trash_ + material value.

                    instance_found = True

                if not instance_found:
                    id_string = 'trash_unknown_instance'
        

    else:
        return None    # If the class title isn't any of the above, just skip.

    return id_string

def process_annotations(file_list, set_type):
    global trash_unknown_akas ,annotation_id, image_id

    annotations = []
    
    images = []

    if dataset_version == 'material':
        category_ids = {
            'rov':                  1,
            'plant':                2,
            'animal_fish':          3,
            'animal_starfish':      4,
            'animal_shells':        5,
            'animal_crab':          6,
            'animal_eel':           7,
            'animal_etc':           8,
            'trash_etc':            9,
            'trash_fabric':         10,
            'trash_fishing_gear':   11,
            'trash_metal':          12,
            'trash_paper':          13,
            'trash_plastic':        14,
            'trash_rubber':         15,
            'trash_wood':           16
        }
    else:
        category_ids = {
            'rov':                      1,
            'plant':                    2,
            'animal_fish':              3,
            'animal_starfish':          4,
            'animal_shells':            5,
            'animal_crab':              6,
            'animal_eel':               7,
            'animal_etc':               8,
            'trash_clothing':           9,
            'trash_pipe':               10,
            'trash_bottle':             11,
            'trash_bag':                12,
            'trash_snack_wrapper':      13,
            'trash_can':                14,
            'trash_cup':                15,
            'trash_container':          16,
            'trash_unknown_instance':   17,
            'trash_branch':             18,
            'trash_wreckage':           19,
            'trash_tarp':               20,
            'trash_rope':               21,
            'trash_net':                22
        }

    obj_dict = dict()
    for ann_file in tqdm(file_list):
        # create annotations for the image
        with open('./annotations/'+ann_file) as json_file:
            data = json.load(json_file)

            width = int(data['size']['width'])
            height = int(data['size']['height'])

            objects = data['objects']
            for obj in objects:
                id_string = get_category_id_string(obj, ann_file)
                if id_string:
                    obj_dict[id_string] = obj_dict.get(id_string, 0) +1
                    category_id = category_ids[id_string]
                else:
                    continue

                add_submasks = [] #List of masks to add on to the object.
                if 'instance' in obj.keys(): #If an object is bound to another, it has an instance tag.
                    #NOTE that this is different from an instance tag in the tags array.
                    instance = obj['instance']
                    for o in objects:
                        if 'instance' in o.keys() and o['instance'] == instance:
                            if o['id'] == obj['id']:
                                continue
                            bmp = base64_2_mask(o['bitmap']['data'])
                            orig = o['bitmap']['origin']
                            add_submasks.append(submask_2_mask(bmp, orig, height, width))
                            objects.remove(o)


                submask = base64_2_mask(obj['bitmap']['data']) #Get bitmap and origin for current object.
                origin = obj['bitmap']['origin']

                mask = submask_2_mask(submask, origin, height, width)
                         
                if len(add_submasks) > 0:
                    # print ("Adding other instance masks, # masks: ", len(add_submasks))
                    # cv2.imshow('original', mask)
                    # cv2.waitKey()
                    for m in add_submasks:
                        # cv2.imshow('added_mask', m)
                        # cv2.waitKey()
                        mask = cv2.bitwise_or(mask, m)

                    # cv2.imshow('final', mask)
                    # cv2.waitKey
                
                annotation = create_mask_annotation(ann_file[:-5], mask, image_id, category_id, annotation_id, is_crowd, False)
                if annotation:
                    annotations.append(annotation)
                    annotation_id += 1
                

            copy('./images/'+ann_file[:-5], './' +dataset_version + '_version/'+ set_type + '/'+ann_file[:-5])
            # create image entry for the image
            images.append(create_image_entry(ann_file[:-5], image_id, width, height))
            image_id += 1


    if dataset_version == 'material':
        #Write materials version
        f = open("./" + dataset_version +"_version/instances_"+set_type+"_trashcan.json", "w")
        json.dump({"info": {
            "description": "TrashCAN Segmentation Dataset",
            "url": "N/A",
            "version": "0.5",
            "year": 2020,
            "contributor": "IRVLab",
            "date_created": "2020/6/24"
        },
        "licenses": [
            {
                "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
                "id": 1,
                "name": "Attribution-NonCommercial-ShareAlike License"
            }
        ],
        "categories": [
            {"supercategory": "rov","id": 1,"name": "rov"},
            {"supercategory": "plant","id": 2,"name": "plant"},
            {"supercategory": "animal_fish","id": 3,"name": "animal_fish"},
            {"supercategory": "animal_starfish","id": 4,"name": "animal_starfish"},
            {"supercategory": "animal_shells","id": 5,"name": "animal_shells"},
            {"supercategory": "animal_crab","id": 6,"name": "animal_crab"},
            {"supercategory": "animal_eel","id": 7,"name": "animal_eel"},
            {"supercategory": "animal_etc","id": 8,"name": "animal_etc"},
            {"supercategory": "trash_etc","id": 9,"name": "trash_etc"},
            {"supercategory": "trash_fabric","id": 10,"name": "trash_fabric"},
            {"supercategory": "trash_fishing_gear","id": 11,"name": "trash_fishing_gear"},
            {"supercategory": "trash_metal","id": 12,"name": "trash_metal"},
            {"supercategory": "trash_paper","id": 13,"name": "trash_paper"},
            {"supercategory": "trash_plastic","id": 14,"name": "trash_plastic"},
            {"supercategory": "trash_rubber","id": 15,"name": "trash_rubber"},
            {"supercategory": "trash_wood","id": 16,"name": "trash_wood"}
        ],"images": images, "annotations": annotations}, f, indent=2)
        f.close()
    else:
        #Write instance version
        f = open("./" + dataset_version +"_version/instances_"+set_type+"_trashcan.json", "w")
        json.dump({"info": {
            "description": "TrashCAN Segmentation Dataset",
            "url": "N/A",
            "version": "0.5",
            "year": 2020,
            "contributor": "IRVLab",
            "date_created": "2020/6/24"
        },
        "licenses": [
            {
                "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
                "id": 1,
                "name": "Attribution-NonCommercial-ShareAlike License"
            }
        ],
        "categories": [
            {"supercategory": "rov","id": 1,"name": "rov"},
            {"supercategory": "plant","id": 2,"name": "plant"},
            {"supercategory": "animal_fish","id": 3,"name": "animal_fish"},
            {"supercategory": "animal_starfish","id": 4,"name": "animal_starfish"},
            {"supercategory": "animal_shells","id": 5,"name": "animal_shells"},
            {"supercategory": "animal_crab","id": 6,"name": "animal_crab"},
            {"supercategory": "animal_eel","id": 7,"name": "animal_eel"},
            {"supercategory": "animal_etc","id": 8,"name": "animal_etc"},
            {"supercategory": "trash_clothing","id": 9,"name": "trash_clothing"},
            {"supercategory": "trash_pipe","id": 10,"name": "trash_pipe"},
            {"supercategory": "trash_bottle","id": 11,"name": "trash_bottle"},
            {"supercategory": "trash_bag","id": 12,"name": "trash_bag"},
            {"supercategory": "trash_snack_wrapper","id": 13,"name": "trash_snack_wrapper"},
            {"supercategory": "trash_can","id": 14,"name": "trash_can"},
            {"supercategory": "trash_cup","id": 15,"name": "trash_cup"},
            {"supercategory": "trash_container","id": 16,"name": "trash_container"},
            {"supercategory": "trash_unknown_instance","id": 17,"name": "trash_unknown_instance"},
            {"supercategory": "trash_branch","id": 18,"name": "trash_branch"},
            {"supercategory": "trash_wreckage","id": 19,"name": "trash_wreckage"},
            {"supercategory": "trash_tarp","id": 20,"name": "trash_tarp"},
            {"supercategory": "trash_rope","id": 21,"name": "trash_rope"},
            {"supercategory": "trash_net","id": 22,"name": "trash_net"}
        ],"images": images, "annotations": annotations}, f, indent=2)
        f.close()

    return obj_dict

if __name__ == '__main__':

    images_list = list() # [img1, img2, im3 ...]
    overall_objects = dict() #{trash:450, animal:300 ...}
    image_object_dict = dict() # {image1: {trash:2, rov:5, animal:1}...}
    object_image_dict = dict() #{trash: [img3, img6, im18], rov:[]...}

    total_imgs = 0

    print("Initial processing step")
    for filename in tqdm(os.listdir('./annotations')):
        if filename.endswith('.jpg.json'):
            total_imgs += 1
            images_list.append(filename)

            with open('./annotations/'+filename) as json_file:
                data = json.load(json_file)
                objects = data['objects']

                obj_dict = dict()

                for obj in objects:
                    id_string = get_category_id_string(obj, filename)
                    if id_string:
                        overall_objects[id_string] = overall_objects.get(id_string, 0) + 1
                        obj_dict[id_string] = obj_dict.get(id_string, 0) +1

                        if not id_string in object_image_dict.keys():
                            object_image_dict[id_string] = list()

                        if not filename in object_image_dict[id_string]:
                            object_image_dict[id_string].append(filename)

                    else:
                        continue
                
                image_object_dict[filename] = obj_dict

    training_names = []
    val_names = []
    current_objs = dict.fromkeys(overall_objects.keys(), 0)

    print ("Creating validation set")
    for id_string in tqdm(sorted(overall_objects, key= lambda key:overall_objects[key])):
        num_objs = overall_objects[id_string]

        if True: #If there is a small enough number of instances
            val_desired = int(num_objs*0.20)
            # print("For " + str(id_string) + " trying for: " + str(val_desired) +" objects...")

            vids_for_val = []
            vids_for_train = []

            swip_swop = True
            images_added = 0
            images_containing = object_image_dict[id_string]

            for image in images_containing:
                if image in images_list:
                    if current_objs[id_string] <= val_desired:
                        if swip_swop:
                            images_list.remove(image) #Remove from overall image list
                            vids_for_val.append(image) # add to validation
    
                            for key in image_object_dict[image].keys(): #Add object counts to current count.
                                current_objs[key] += image_object_dict[image][key]

                            # Count up to 10 images, then switch to train
                            images_added += 1
                            if images_added ==10:
                                images_added = 0
                                swip_swop = not swip_swop

                        else:
                            images_list.remove(image) #Remove from overall image list
                            vids_for_train.append(image) # add to training

                            images_added += 1
                            if images_added ==10:
                                images_added = 0
                                swip_swop = not swip_swop


                # if video in video_images_dict:
                #     if current_objs[id_string] <= val_desired:
                #         # print("Currently " + str(current_objs[id_string]) +  " " + str(id_string) + " adding " + str(video_objects_dict[video][id_string]) + " objects...")    
                #         for key in video_objects_dict[video].keys():
                #             current_objs[key] += video_objects_dict[video][key]

                #         vids_for_val.extend(video_images_dict.pop(video))
                #     else:
                #         continue
                
        
            val_names.extend(vids_for_val)
            training_names.extend(vids_for_train)

        else:
            continue

    training_names.extend(images_list)

    random.shuffle(training_names)
    random.shuffle(val_names)

    print("Processing training images")
    train_obj_dict = process_annotations(training_names, "train")
    # print train_obj_dict

    print("Processing validation images")
    val_obj_dict = process_annotations(val_names, "val")
    # print val_obj_dict

    # for fail in sorted(bio_fail):
        # print(fail)


    print("Obj summary(" + dataset_version +'):')
    print('|Class Name  | Train #   | Val # | Total #   | Percentage (Train -- Val) |')
    print('|:---------- |:---------:|:-----:|:---------:|:----------:|')
    for key in sorted(overall_objects):
        percent = "     | {:.2f} -- {:.2f} |".format(float(train_obj_dict[key])/float(overall_objects[key]) ,float(val_obj_dict[key])/float(overall_objects[key]))
        
        print( "| " + key + "    | " + str(train_obj_dict[key]) + "   | " + str(val_obj_dict[key]) + "     | " + str(overall_objects[key]) + percent ) 
    
    percent = "  Percentage: {:.2f}".format((float(len(training_names))/float(total_imgs)))
    print("\nTrain size: " + str(len(training_names)) + percent)

    percent = "  Percentage: {:.2f}".format((float(len(val_names))/float(total_imgs)))
    print("Val size: " + str(len(val_names)) + percent)

    