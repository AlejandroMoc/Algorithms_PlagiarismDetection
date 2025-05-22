from pypdf import PdfReader, PdfWriter
import os

def combinar_archivos_pdf(archivos_pdf, ruta_salida):
    """Combina varios archivos PDF en uno solo."""
    if not archivos_pdf:
        print("No se han proporcionado archivos PDF para combinar.")
        return False

    escritor = PdfWriter()
    
    try:
        for pdf in archivos_pdf:
            if not os.path.isfile(pdf) or not pdf.lower().endswith('.pdf'):
                print(f"Saltando archivo PDF inválido: {pdf}")
                continue
            
            lector = PdfReader(pdf)
            for pagina in lector.pages:
                escritor.add_page(pagina)
        
        with open(ruta_salida, "wb") as archivo_salida:
            escritor.write(archivo_salida)
        return True
    except Exception as e:
        print(f"Error al combinar los PDFs: {str(e)}")
        return False

def combinar_archivos_pdf_con_paginas(configuracion_pdf, ruta_salida):
    """Combina páginas específicas de varios PDFs."""
    if not configuracion_pdf:
        print("No se han proporcionado configuraciones de PDF para combinar.")
        return False

    escritor = PdfWriter()
    
    try:
        for ruta_pdf, paginas in configuracion_pdf:
            if not os.path.isfile(ruta_pdf) or not ruta_pdf.lower().endswith('.pdf'):
                print(f"Saltando archivo PDF inválido: {ruta_pdf}")
                continue

            lector = PdfReader(ruta_pdf)
            total_paginas = len(lector.pages)
            
            if paginas is not None:
                for num_pagina in paginas:
                    if 0 <= num_pagina < total_paginas:
                        escritor.add_page(lector.pages[num_pagina])
                    else:
                        print(f"Página {num_pagina} fuera de rango para el archivo {ruta_pdf}. Saltando.")
            else:
                for pagina in lector.pages:
                    escritor.add_page(pagina)
        
        with open(ruta_salida, "wb") as archivo_salida:
            escritor.write(archivo_salida)
        return True
    except Exception as e:
        print(f"Error al combinar los PDFs: {str(e)}")
        return False