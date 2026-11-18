"""Preparación inicial completa; las siguientes aperturas reutilizan los archivos."""
from .config import DATA_DIR,AUDIO_DIR,METADATA_DIR,AUDIO_CACHE
from .prepare import prepare
from .prepare_multimodal import prepare_space,prepare_text,prepare_audio
from .minidb.multimodal import MultimodalDatabase


def main():
    from data.prepare_fma import ensure_data
    ensure_data()
    prepare(DATA_DIR)
    db=MultimodalDatabase(DATA_DIR)
    try:
        prepare_space(db)
        prepare_text(db,METADATA_DIR)
        if 'pistas' not in db.catalog:
            from .audio.prepare import prepare as extract
            print('Primera preparación: extraer 8000 pistas puede tardar una hora o más. El proceso es reanudable.',flush=True)
            extract(AUDIO_DIR,METADATA_DIR,AUDIO_CACHE,limit=None,resume=True,device='cpu',batch_size=2)
        prepare_audio(db,AUDIO_CACHE)
        print('SonoraDB está preparado. Abrir http://127.0.0.1:8014',flush=True)
    finally:
        db.close()


if __name__=='__main__':
    main()
