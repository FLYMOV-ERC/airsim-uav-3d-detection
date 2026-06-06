#!/usr/bin/env python3
"""
Captura simples e direta - sem complicações
"""

import airsim
import time

print("📸 CAPTURA SIMPLES E DIRETA\n")

# Conecta
client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
client.confirmConnection()
print("✅ Conectado!\n")

# Captura 10 imagens rapidamente
for i in range(10):
    try:
        # Captura com front_center que funciona
        img = client.simGetImage("front_center", airsim.ImageType.Scene, "Ego")

        if img and len(img) > 1000:
            filename = f"captura_{i:03d}.png"
            with open(filename, 'wb') as f:
                f.write(img)
            print(f"✅ Salvou {filename} ({len(img)/1024:.0f}KB)")
        else:
            print(f"❌ Frame {i} - sem dados")

    except Exception as e:
        print(f"❌ Erro no frame {i}: {e}")

    time.sleep(1)

print(f"\n✅ Completo! Verifique as imagens captura_*.png")