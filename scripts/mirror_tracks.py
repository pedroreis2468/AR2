import os
import yaml
import glob

def mirror_track(input_path, output_path):
    with open(input_path, 'r') as f:
        data = yaml.safe_load(f)

    track = data.get('track', {})
    if not track:
        return

    # Função auxiliar para inverter o Y e mudar a cor do cone
    def mirror_points(points_list, new_class):
        new_list = []
        for p in points_list:
            new_p = p.copy()
            new_p['position'][1] = -float(new_p['position'][1]) # Inverte o Y
            
            # Mantém classes especiais (big-orange, etc) ou força a cor base
            if 'orange' not in new_p.get('class', ''):
                new_p['class'] = new_class
                
            new_list.append(new_p)
        return new_list

    # Guarda as originais
    orig_left = track.get('left', [])
    orig_right = track.get('right', [])

    # Troca: Os cones da direita originais passam a ser a esquerda (Azul)
    track['left'] = mirror_points(orig_right, 'blue')
    # Os cones da esquerda originais passam a ser a direita (Amarelo)
    track['right'] = mirror_points(orig_left, 'yellow')

    # Espelhar timekeeping e unknown
    if 'time_keeping' in track:
        track['time_keeping'] = mirror_points(track['time_keeping'], 'timekeeping')
    if 'unknown' in track:
        track['unknown'] = mirror_points(track['unknown'], 'unknown')

    # Inverter o ângulo de yaw inicial (se o carro começasse virado)
    if 'start' in track and 'orientation' in track['start']:
        track['start']['orientation'][2] = -float(track['start']['orientation'][2])

    with open(output_path, 'w') as f:
        yaml.safe_dump(data, f, sort_keys=False)

def main():
    # Cria a pasta para as pistas espelhadas
    out_dir = os.path.join('tracks', 'mirrored')
    os.makedirs(out_dir, exist_ok=True)
    
    # Procura todas as pistas yaml na pasta atual ou tracks/
    # Ajusta o caminho se as tuas pistas estiverem noutra pasta
    track_files = glob.glob('tracks/*.yaml') 
    
    for file_path in track_files:
        filename = os.path.basename(file_path)
        # Ignora ficheiros que não são pistas
        if filename in ['gripMap.yaml', 'skidpad.yaml', 'acceleration.yaml']:
            continue
            
        out_path = os.path.join(out_dir, f"mirrored_{filename}")
        mirror_track(file_path, out_path)
        print(f"[OK] Criada pista espelhada: {out_path}")

if __name__ == '__main__':
    main()