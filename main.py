from update_curve import update_curve
from calc_percentile import calc_percentile
from mask_to_volume import mask_to_volume


def main(MA,masks_path):
    current_volume = mask_to_volume(masks_path)
    print("Choose option:\n1 - update percentile curves\n2 - calculate the volume's percentile")
    options = input('your choice is: ')
    if options == '1':
        update_curve()
    elif options == '2':
        calc_percentile(MA, current_volume)
    else:
        print("option doesn't exist")
        main(MA,masks_path)
path=r'C:\Users\User\Google Drive\TAU\4th year\final project\firstAssignment\masks\mask_first_try\masks.mat'
main(23,path)
