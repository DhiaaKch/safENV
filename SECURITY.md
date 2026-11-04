# Configuration de développement locale uniquement

Le stack Compose de démonstration désactive l'authentification Elasticsearch. Ses ports publiés sont désormais liés à `127.0.0.1` : ne retirez pas cette restriction sur une machine accessible depuis le réseau. N'utilisez pas ces paramètres avec des données sensibles ou en production. Pour un déploiement, activez l'authentification et TLS, utilisez des secrets dédiés hors du dépôt, et limitez l'accès réseau aux services.
