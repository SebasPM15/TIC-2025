import os

# Directorio raíz del dataset
sync_dir = "/home/lasinac/ticDSO/Implementacion paper 9/DCDepth/sync"
list_file = "/home/lasinac/ticDSO/Implementacion paper 9/DCDepth/data_splits/nyudepthv2_test_files_with_gt.txt"
output_file = "/home/lasinac/ticDSO/Implementacion paper 9/DCDepth/data_splits/nyudepthv2_test_files_with_gt_updated.txt"

# Leer el archivo original
with open(list_file, 'r') as f:
    lines = f.readlines()

# Abrir archivo de salida
with open(output_file, 'w') as f_out:
    for line in lines:
        # Parsear la línea
        parts = line.strip().split()
        if len(parts) != 3:
            print(f"Línea inválida: {line.strip()}")
            continue

        rgb_path, depth_path, focal = parts
        rgb_filename = os.path.basename(rgb_path)  # Ejemplo: rgb_00045.jpg
        depth_filename = os.path.basename(depth_path)  # Ejemplo: sync_depth_00045.png

        # Obtener la categoría (ejemplo: "bathroom")
        category = rgb_path.split('/')[0]

        # Buscar la imagen en las subcarpetas de sync que comiencen con la categoría
        found_rgb = False
        found_depth = False
        new_rgb_path = None
        new_depth_path = None

        for subdir in os.listdir(sync_dir):
            if subdir.startswith(category):
                subdir_path = os.path.join(sync_dir, subdir)
                if os.path.isdir(subdir_path):
                    # Buscar la imagen RGB
                    potential_rgb = os.path.join(subdir_path, rgb_filename)
                    if os.path.exists(potential_rgb):
                        new_rgb_path = os.path.join(subdir, rgb_filename)
                        found_rgb = True

                    # Buscar la imagen de profundidad
                    potential_depth = os.path.join(subdir_path, depth_filename)
                    if os.path.exists(potential_depth):
                        new_depth_path = os.path.join(subdir, depth_filename)
                        found_depth = True

                    if found_rgb and found_depth:
                        break

        if found_rgb and found_depth:
            # Escribir la línea actualizada
            f_out.write(f"{new_rgb_path} {new_depth_path} {focal}\n")
        else:
            print(f"No se encontraron los archivos: {rgb_path}, {depth_path}")