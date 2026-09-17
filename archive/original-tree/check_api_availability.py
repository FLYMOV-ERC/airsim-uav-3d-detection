#!/usr/bin/env python3
"""
Verifica quais APIs estão disponíveis no cosysairsim
"""

import cosysairsim as airsim
import sys

print("\n" + "="*70)
print("🔍 VERIFICANDO APIs DISPONÍVEIS NO COSYSAIRSIM")
print("="*70)

# Conecta
try:
    client = airsim.MultirotorClient(ip="172.19.80.1", port=41451)
    client.confirmConnection()
    print("✅ Conectado ao AirSim")
except Exception as e:
    print(f"❌ Erro conectando: {e}")
    sys.exit(1)

# Lista todas as funções do client
all_methods = dir(client)

print(f"\n📊 Total de métodos no client: {len(all_methods)}")

# Categoriza os métodos
categories = {
    "Detection API": ["detect", "Detection"],
    "Segmentation API": ["segment", "Segmentation"],
    "Mesh API": ["mesh", "Mesh"],
    "Object API": ["object", "Object"],
    "Get APIs": ["simGet"],
    "Set APIs": ["simSet"],
    "Clear APIs": ["simClear"],
    "Add APIs": ["simAdd"]
}

for category, keywords in categories.items():
    print(f"\n📁 {category}:")
    found = False
    for method in all_methods:
        for keyword in keywords:
            if keyword in method:
                print(f"   - {method}")
                found = True
                break
    if not found:
        print("   (nenhum método encontrado)")

# Verifica especificamente a API de detecção
print("\n🎯 VERIFICAÇÃO ESPECÍFICA DA API DE DETECÇÃO:")

detection_methods = [
    'simGetDetections',
    'simSetDetectionFilterRadius',
    'simAddDetectionFilterMeshName',
    'simClearDetectionMeshNames',
    'simGetMeshPositionVertexBuffers'
]

api_available = True
for method in detection_methods:
    exists = hasattr(client, method)
    status = "✅" if exists else "❌"
    print(f"   {status} {method}: {'DISPONÍVEL' if exists else 'NÃO DISPONÍVEL'}")
    if not exists:
        api_available = False

if not api_available:
    print("\n⚠️ PROBLEMA: A API de detecção NÃO está completamente disponível!")
    print("   O módulo cosysairsim pode não ter essa funcionalidade.")
    print("\n   SOLUÇÕES:")
    print("   1. Usar o módulo 'airsim' oficial ao invés de 'cosysairsim'")
    print("   2. Atualizar ou recompilar o cosysairsim com suporte a detecção")
    print("   3. Usar método alternativo (posições conhecidas dos drones)")
else:
    print("\n✅ API de detecção está disponível!")

# Tenta importar o airsim oficial para comparar
print("\n🔄 Tentando importar módulo 'airsim' oficial para comparação...")
try:
    import airsim as official_airsim
    official_client = official_airsim.MultirotorClient()

    print("✅ Módulo airsim oficial importado!")

    # Compara APIs
    official_methods = set(dir(official_client))
    cosys_methods = set(all_methods)

    only_in_official = official_methods - cosys_methods
    only_in_cosys = cosys_methods - official_methods

    if only_in_official:
        print(f"\n📊 Métodos APENAS no airsim oficial ({len(only_in_official)}):")
        for method in list(only_in_official)[:10]:  # Mostra até 10
            print(f"   - {method}")
        if len(only_in_official) > 10:
            print(f"   ... e mais {len(only_in_official) - 10} métodos")

    # Verifica se o oficial tem a API de detecção
    print("\n🎯 API de detecção no módulo OFICIAL:")
    for method in detection_methods:
        exists = hasattr(official_client, method)
        status = "✅" if exists else "❌"
        print(f"   {status} {method}")

except ImportError:
    print("❌ Módulo airsim oficial não está instalado")
    print("   Para instalar: pip install airsim")
except Exception as e:
    print(f"⚠️ Erro ao testar módulo oficial: {e}")

print("\n" + "="*70)