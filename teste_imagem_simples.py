#!/usr/bin/env python3
import airsim
import time

print("📸 TESTE RÁPIDO DE CAPTURA DE IMAGEM - MOUNTAINS\n")

client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
client.confirmConnection()
print("✅ Conectado!")

# Arma o Ego
client.enableApiControl(True, "Ego")
client.armDisarm(True, "Ego")

# Decola
print("🛫 Decolando...")
client.takeoffAsync(vehicle_name="Ego").join()
time.sleep(3)

# Tenta capturar imagem
print("\n📸 Tentando capturar imagem...")

try:
    # Método mais simples - imagem PNG direta
    png = client.simGetImage("0", airsim.ImageType.Scene, vehicle_name="Ego")

    if png and len(png) > 0:
        with open("teste_mountains.png", 'wb') as f:
            f.write(png)
        print(f"✅ IMAGEM SALVA! ({len(png)} bytes)")
        print("📁 Arquivo: teste_mountains.png")
    else:
        print("❌ Nenhuma imagem capturada")
        print("\n⚠️ PROBLEMA: O ambiente Mountains pode não ter câmeras configuradas")
        print("Você precisa adicionar câmeras no settings.json")

except Exception as e:
    print(f"❌ Erro: {e}")
    print("\n⚠️ O settings.json precisa ter câmeras configuradas para o drone Ego")

# Pousa
print("\n🛬 Pousando...")
client.landAsync(vehicle_name="Ego").join()
client.armDisarm(False, "Ego")
client.enableApiControl(False, "Ego")

print("\n💡 DICA: Se não capturou imagem, adicione isto ao settings.json no Windows:")