from options.config import get_arguments
from utils.manipulate import *
from models.training import *
from utils.imresize import imresize
from utils.imresize import imresize_to_shape
import utils.functions as functions

#运行代码：python test.py --input_name countryside --test_name countryside_test --paint_start_scale 1


if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/training_pair')
    parser.add_argument('--input_name', help='training image name', required=True)
    parser.add_argument('--test_dir', help='input test dir', default='Input/test')
    parser.add_argument('--test_name', help='test image name', required=True)
    parser.add_argument('--paint_start_scale', help='paint injection scale', type=int, required=True)
    parser.add_argument('--quantization_flag', help='specify if to perform color quantization training', type=bool, default=False)
    parser.add_argument('--mode', help='task to be done', default='test') # 设 mode
    opt = parser.parse_args()
    opt = functions.post_config(opt)
    Gs = []
    Zs = []
    images2 = []
    test_images = []
    NoiseAmp = []
    dir2save = functions.generate_dir2save(opt)#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if dir2save is None:
        print('task does not exist')
    # elif (os.path.exists(dir2save)):
    #     print("output already exist")
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass
        image2 = functions.read_target_image(opt) #读入原始image2（在这个函数中只是用来为test_images参考，做规格修改）
        image2 = functions.adjust_scales2image(image2, opt) # 根据原始image2决定scale
        Gs, Zs, images2, NoiseAmp = functions.load_trained_pyramid(opt)

        if (opt.paint_start_scale < 1) | (opt.paint_start_scale > (len(Gs)-1)):
            print("injection scale should be between 1 and %d" % (len(Gs)-1))
        else: # 读取test_image
            test_image = functions.read_image_dir('%s/%s' % (opt.test_dir, opt.test_name), opt)
            if test_image.shape[3] != image2.shape[3]:
                test_image = imresize_to_shape(test_image, [image2.shape[2], image2.shape[3]], opt)
                test_image = test_image[:, :, :image2.shape[2], :image2.shape[3]]
            #修改test_image并创建pyramid
            test_images = functions.creat_reals_pyramid(test_image, test_images, opt)
            #下面是下采样到最低层，然后上采样到N-1层
            N = len(images2) - 1
            n = opt.paint_start_scale
            in_s = imresize(test_image, pow(opt.scale_factor, (N - n + 1)), opt)
            in_s = in_s[:, :, :images2[n - 1].shape[2], :images2[n - 1].shape[3]]
            in_s = imresize(in_s, 1 / opt.scale_factor, opt)
            in_s = in_s[:, :, :images2[n].shape[2], :images2[n].shape[3]]

            # if opt.quantization_flag:
            #     opt.mode = 'paint_train'
            #     dir2trained_model = functions.generate_dir2save(opt)
            #     # N = len(reals) - 1
            #     # n = opt.paint_start_scale
            #     real_s = imresize(real, pow(opt.scale_factor, (N - n)), opt)
            #     real_s = real_s[:, :, :reals[n].shape[2], :reals[n].shape[3]]
            #     real_quant, centers = functions.quant(real_s, opt.device)
            #     plt.imsave('%s/real_quant.png' % dir2save, functions.convert_image_np(real_quant), vmin=0, vmax=1)
            #     plt.imsave('%s/in_paint.png' % dir2save, functions.convert_image_np(in_s), vmin=0, vmax=1)
            #     in_s = functions.quant2centers(ref, centers)
            #     in_s = imresize(in_s, pow(opt.scale_factor, (N - n)), opt)
            #     # in_s = in_s[:, :, :reals[n - 1].shape[2], :reals[n - 1].shape[3]]
            #     # in_s = imresize(in_s, 1 / opt.scale_factor, opt)
            #     in_s = in_s[:, :, :reals[n].shape[2], :reals[n].shape[3]]
            #     plt.imsave('%s/in_paint_quant.png' % dir2save, functions.convert_image_np(in_s), vmin=0, vmax=1)
            #     if (os.path.exists(dir2trained_model)):
            #         # print('Trained model does not exist, training SinGAN for SR')
            #         Gs, Zs, reals, NoiseAmp = functions.load_trained_pyramid(opt)
            #         opt.mode = 'paint2image'
            #     else:
            #         train_paint(opt, Gs, Zs, reals, NoiseAmp, centers, opt.paint_start_scale)
            #         opt.mode = 'paint2image'
            #out = generate(Gs[n:], Zs[n:], reals, NoiseAmp[n:], opt, in_s, n=n, num_samples=1)
            # plt.imsave('%s/in_s.png' % (dir2save), functions.convert_image_np(in_s.detach()), vmin=0, vmax=1)
            out = generate(Gs[n:], Zs[n:], test_images[n:], images2, NoiseAmp[n:], opt, in_s, n=n,  num_samples=1)
            #plt.imsave('%s/start_scale=%d.png' % (dir2save, opt.paint_start_scale), functions.convert_image_np(out.detach()), vmin=0, vmax=1)
            plt.imsave('%s/%s.png' % (dir2save, opt.test_name), functions.convert_image_np(out.detach()), vmin=0, vmax=1)





