import argparse
import UseTrainedModel as tm


argparser=argparse.ArgumentParser()
argparser.add_argument("imagepath")
argparser.add_argument("checkpoint")
argparser.add_argument('----top_k',help='Return top KK most likely classes', type=int,default=3)
argparser.add_argument('--category_names',help='Use a mapping of categories to real names',  type=str,default="cat_to_name.json")
argparser.add_argument('--gpu',action="store_true", help='Use GPU for inference')

args=argparser.parse_args()
#print(args)
#print(args.imagepath)
train=tm.UseTrainedModel()
train.make_predictions(args.imagepath,args.checkpoint,args.top_k,args.category_names,args.gpu)

