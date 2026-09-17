# 🗺️ Ambientes 3D do AirSim - Como Instalar e Usar

## ✅ O QUE VOCÊ JÁ TEM:
- **Blocks**: Ambiente básico com blocos (já está rodando)

## 🌍 AMBIENTES DISPONÍVEIS PARA DOWNLOAD:

### 1. **AirSimNH (Neighborhood)** - Bairro Suburbano
- **Download**: https://github.com/Microsoft/AirSim/releases
- **Tamanho**: ~500 MB
- **Cenário**: Casas, ruas, árvores

### 2. **City** - Cidade
- **Download**: Via Unreal Marketplace
- **Cenário**: Prédios, ruas urbanas

### 3. **Mountains** - Montanhas
- **Download**: AirSim releases
- **Cenário**: Terreno montanhoso, lagos

### 4. **Africa** - Savana Africana
- **Download**: Epic Games Launcher
- **Cenário**: Savana, animais, vegetação

## 📥 COMO INSTALAR NOVOS AMBIENTES:

### Opção 1: Ambientes Prontos (Mais Fácil)

1. **Baixe o ambiente** dos links acima
2. **Extraia** o arquivo .zip
3. **Execute** o .exe (Windows)
4. **Use o mesmo settings.json** - funciona em todos!

### Opção 2: Criar seu Próprio (Avançado)

1. Instale **Unreal Engine 4.27**
2. Instale o **plugin AirSim**
3. Crie ou baixe um mapa do Marketplace
4. Compile o projeto

## 🔄 COMO TROCAR ENTRE AMBIENTES:

### Método 1: Manual (um por vez)
```bash
# Fecha o Blocks.exe
# Abre o AirSimNH.exe (ou outro)
# Roda seu script Python - detecta automaticamente!
```

### Método 2: Automatizado (múltiplos)
```python
import subprocess
import time

ambientes = [
    "C:/AirSim/Blocks/Blocks.exe",
    "C:/AirSim/Neighborhood/AirSimNH.exe",
    "C:/AirSim/Mountains/Mountains.exe"
]

for ambiente in ambientes:
    # Inicia ambiente
    processo = subprocess.Popen(ambiente)
    time.sleep(30)  # Espera carregar

    # Roda coleta de dados
    subprocess.run(["python", "collect-dataset.py"])

    # Fecha ambiente
    processo.terminate()
    time.sleep(5)
```

## ⚠️ LIMITAÇÕES:

1. **Não dá para mudar o mapa 3D em tempo real** - precisa reiniciar
2. **Cada ambiente = arquivo executável separado**
3. **Ambientes pesados** - alguns têm vários GB

## 💡 ALTERNATIVAS SEM BAIXAR NOVOS AMBIENTES:

### No mesmo Blocks você pode:

1. **Mudar drasticamente o visual:**
```python
# Neblina densa = parece outro lugar
client.simSetWeatherParameter(airsim.WeatherParameter.Fog, 1.0)

# Noite = completamente diferente
client.simSetTimeOfDay(True, "2024-01-01 22:00:00")

# Chuva forte
client.simSetWeatherParameter(airsim.WeatherParameter.Rain, 1.0)
```

2. **Spawnar objetos** (se configurado no Unreal):
```python
# Alguns ambientes permitem adicionar objetos
client.simSpawnObject("obstacle1", "Cylinder",
    airsim.Pose(airsim.Vector3r(10, 10, 0)))
```

3. **Usar diferentes altitudes:**
- Voar baixo = vê mais detalhes
- Voar alto = parece cenário diferente

## 📊 RESUMO:

- **Cenários de MOVIMENTO**: ✅ Já funcionam (formações, padrões)
- **Mudanças de CLIMA/HORÁRIO**: ✅ Já funcionam
- **Ambientes 3D DIFERENTES**: ❌ Precisa baixar/instalar separadamente
- **Trocar ambientes**: ⚠️ Precisa reiniciar o simulador