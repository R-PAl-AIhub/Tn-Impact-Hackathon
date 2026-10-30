def is_overlap(boxA,boxB):
    xA = max(boxA[0],boxB[0])
    yA = max(boxA[1],boxB[1])
    xB = min(boxA[2],boxB[2])
    yB = min(boxA[3],boxB[3])
    return xA < xB and yA < yB


#A helmet belongs to a person if the helmet’s center lies inside the person’s bounding box AND is closer to the top than the bottom
def object_belongs_to_person(person_box,helmet_box):
    px1,py1,px2,py2 = person_box
    hx1,hy1,hx2,hy2 = helmet_box

    hx_c = (hx1+hx2)/2
    hy_c = (hy1 + hy2) / 2

    if not (px1 <= hx_c <= px2 and py1 <= hy_c <= py2):
        return False
    
    return (hy_c - py1) <= (py2 - hy_c)


