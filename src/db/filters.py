doesnt_exists_or_0 = {"$in": [0, None]}

material_filter = {
    "has_impregnated_metal": False, 
    "sup_Zn": doesnt_exists_or_0, 
    "sup_Nd": doesnt_exists_or_0, 
    "sup_Sn": doesnt_exists_or_0,
    "ageing_temp": doesnt_exists_or_0,
    "ageing_time": doesnt_exists_or_0,
}
experiment_filter = {
    "H2O_ppm": doesnt_exists_or_0,
    "SO2_ppm": doesnt_exists_or_0
}