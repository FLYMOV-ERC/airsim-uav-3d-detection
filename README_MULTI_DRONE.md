# Sistema de Coleta de Dataset Multi-Drone com Cenários Variados

Este sistema permite criar datasets com múltiplos drones (4+) e cenários variados para treinar modelos de visão computacional e detecção de objetos.

## 🚀 Recursos Implementados

### 1. **Múltiplos Drones (5 total)**
- 1 drone observador principal ("Ego")
- 4 drones intrusos ("Intruder1" a "Intruder4")
- Sistema de anti-colisão automático
- Posicionamento inicial espaçado

### 2. **Padrões de Movimento Diversos**
- **Formação V**: Drones voam em formação V atrás do Ego
- **Órbita Circular**: Movimento circular ao redor do Ego
- **Enxame Aleatório**: Movimento aleatório controlado
- **Perseguição e Evasão**: Pares de drones se perseguem
- **Padrão Misto**: Combinação de diferentes movimentos
- **Figura 8**: Padrão em forma de 8

### 3. **Variações Ambientais**

#### Condições Climáticas:
- Céu limpo
- Neblina leve/pesada
- Chuva leve/pesada
- Neve leve/pesada
- Poeira
- Outono (folhas)

#### Horários do Dia:
- Amanhecer
- Manhã
- Meio-dia
- Tarde
- Hora dourada
- Pôr do sol
- Noite

### 4. **Automação Multi-Ambiente**
Script para executar coleta em diferentes mapas do AirSim automaticamente.

## 📁 Arquivos Criados

1. **`settings.json`** - Configuração do AirSim com 5 drones
2. **`scene_variation.py`** - Módulo para controle de clima e iluminação
3. **`collect-dataset-multi.py`** - Script principal de coleta multi-drone
4. **`run_multi_environments.py`** - Automação para múltiplos ambientes

## 🎮 Como Usar

### 1. Configuração Inicial

Copie o arquivo `settings.json` para o diretório do AirSim:
- Windows: `C:\Users\[seu_usuario]\Documents\AirSim\`
- Linux: `~/Documents/AirSim/`
- macOS: `~/Documents/AirSim/`

### 2. Coleta Básica (Um Ambiente)

```bash
# Coleta simples sem variações
python collect-dataset-multi.py --frames 1000

# Com variações de clima e horário
python collect-dataset-multi.py --frames 1000 --vary_weather --vary_time

# Especificar intervalo de mudanças
python collect-dataset-multi.py --frames 2000 \
    --vary_weather --weather_interval 200 \
    --vary_time --time_interval 300
```

### 3. Coleta em Múltiplos Ambientes

```bash
# Buscar e executar em todos os ambientes encontrados
python run_multi_environments.py \
    --env_dir /path/to/airsim/environments \
    --frames 500

# Especificar ambientes específicos
python run_multi_environments.py \
    --environments Blocks City Mountains \
    --frames 800
```

### 4. Parâmetros Disponíveis

#### `collect-dataset-multi.py`:
- `--frames`: Número de frames para capturar (padrão: 1000)
- `--out`: Diretório de saída (padrão: dataset_multi)
- `--vary_weather`: Ativa variações climáticas
- `--vary_time`: Ativa variações de horário
- `--weather_interval`: Frames entre mudanças de clima (padrão: 200)
- `--time_interval`: Frames entre mudanças de horário (padrão: 300)
- `--intruders`: Lista de nomes dos drones intrusos
- `--cams`: Câmeras para capturar (padrão: front_center)

#### `run_multi_environments.py`:
- `--env_dir`: Diretório com ambientes AirSim
- `--search_paths`: Caminhos adicionais para buscar ambientes
- `--output`: Diretório base de saída
- `--frames`: Frames por ambiente
- `--environments`: Ambientes específicos para usar

## 📊 Estrutura do Dataset Gerado

```
dataset_multi/
├── calibration.json          # Calibração das câmeras
├── images/
│   └── front_center/
│       ├── 000001.png
│       ├── 000002.png
│       └── ...
├── seg/                      # Imagens de segmentação
│   └── front_center/
│       └── ...
├── lidar/                    # Dados LiDAR
│   ├── 000001.npy
│   └── ...
└── meta/                     # Metadados por frame
    ├── 000001.json           # Posições, clima, horário, padrão de movimento
    └── ...
```

### Formato dos Metadados

Cada arquivo em `meta/` contém:
```json
{
  "frame": 1,
  "timestamp": 1234567890.123,
  "vehicles": {
    "Ego": {"pose": {...}},
    "Intruder1": {"pose": {...}},
    ...
  },
  "conditions": {
    "weather": "light_fog",
    "time": "sunset"
  },
  "movement_pattern": "circular_orbit"
}
```

## 🌍 Ambientes Suportados

O sistema procura automaticamente por:
- Blocks
- City
- Neighborhood
- Mountains
- Africa
- AbandonedPark

Baixe ambientes adicionais em: https://github.com/microsoft/AirSim/releases

## 🔧 Requisitos

- Python 3.7+
- AirSim (ou cosysairsim)
- OpenCV (`pip install opencv-python`)
- NumPy
- tqdm

## 💡 Dicas de Uso

1. **Performance**: Para datasets grandes, use `--weather_interval` e `--time_interval` maiores (300-500 frames)

2. **Variedade**: Deixe o sistema rodar por pelo menos 1000 frames para capturar boa diversidade

3. **Múltiplos Ambientes**: Use o script de automação para coletar automaticamente em vários mapas

4. **Personalização**: Edite os padrões de movimento em `DroneMovementPatterns` para criar novos comportamentos

5. **Debugging**: Use `--log_every 10` para ver logs frequentes do progresso

## 🐛 Troubleshooting

**Problema**: "Drone não encontrado"
- **Solução**: Verifique se os nomes no `settings.json` correspondem aos usados no script

**Problema**: "Colisões frequentes"
- **Solução**: Aumente `min_separation` no `MultiDroneController` (padrão: 10m)

**Problema**: "Clima não muda"
- **Solução**: Nem todos os ambientes suportam todos os efeitos. Teste diferentes combinações.

**Problema**: "Ambiente não inicia"
- **Solução**: Verifique o caminho do executável e se tem permissões de execução

## 📈 Melhorias Futuras

- [ ] Adicionar mais padrões de movimento (espiral, zigue-zague)
- [ ] Implementar mudanças graduais de clima
- [ ] Suporte para diferentes tipos de veículos (carros + drones)
- [ ] Exportação para formato COCO/YOLO
- [ ] GUI para configuração visual

## 🎯 Exemplo de Uso Completo

```bash
# 1. Configurar AirSim
cp settings.json ~/Documents/AirSim/

# 2. Iniciar ambiente AirSim (Blocks, City, etc.)

# 3. Coletar dataset com todas as variações
python collect-dataset-multi.py \
    --frames 2000 \
    --out my_dataset \
    --vary_weather \
    --vary_time \
    --weather_interval 150 \
    --time_interval 200 \
    --log_every 50

# 4. Ou usar automação para múltiplos ambientes
python run_multi_environments.py \
    --env_dir ~/AirSimEnvironments \
    --output datasets_all \
    --frames 500 \
    --environments Blocks City Mountains
```

## 📝 Notas

- O sistema foi projetado para ser modular e extensível
- Todos os parâmetros podem ser ajustados conforme necessário
- O código usa threading para movimento contínuo dos drones
- Anti-colisão automática previne acidentes entre drones