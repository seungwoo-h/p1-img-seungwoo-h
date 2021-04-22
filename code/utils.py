from torchvision import transforms

def get_default_transform():
    transform = transforms.Compose([transforms.CenterCrop(350),
                                    transforms.Resize((224, 224)), 
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.548, 0.504, 0.479), 
                                                         (0.237, 0.247, 0.246)),])
    return transform                                        

def label_class(x):
    """
    # mask - 0: normal / 1: mask / 2: incorrect
    # gender - 0: female / 1: male
    # age - <30: 0 / 30< <60: 1 / 60<: 2
    """
    # not wear
    if x[0] == '0':
        if x[1] == '0':
            if x[2] == '0':
                return 15
            elif x[2] == '1':
                return 16
            elif x[2] == '2':
                return 17
        elif x[1] == '1':
            if x[2] == '0':
                return 12
            elif x[2] == '1':
                return 13
            elif x[2] == '2':
                return 14
    # wear
    elif x[0] == '1':
        if x[1] == '0':
            if x[2] == '0':
                return 3
            elif x[2] == '1':
                return 4
            elif x[2] == '2':
                return 5
        elif x[1] == '1':
            if x[2] == '0':
                return 0
            elif x[2] == '1':
                return 1
            elif x[2] == '2':
                return 2
    # incorrect
    elif x[0] == '2':
        if x[1] == '0':
            if x[2] == '0':
                return 9
            elif x[2] == '1':
                return 10
            elif x[2] == '2':
                return 11
        elif x[1] == '1':
            if x[2] == '0':
                return 6
            elif x[2] == '1':
                return 7
            elif x[2] == '2':
                return 8
