"""Institutional report layout preserving the original midterm typography."""
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                              TableStyle, PageBreak, Image)

ROOT = Path(__file__).resolve().parents[1]
R = ROOT/'results'
F = ROOT/'report'/'figures'
NAVY = colors.HexColor('#173c60')
GRAY = colors.HexColor('#566574')
LIGHT = colors.HexColor('#edf3f7')
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyM', fontName='Times-Roman',fontSize=10.6,leading=14.3,
    spaceAfter=8,alignment=TA_JUSTIFY))
styles.add(ParagraphStyle(name='HeadM',fontName='Helvetica-Bold',fontSize=16,leading=20,
    textColor=NAVY,spaceBefore=0,spaceAfter=13))
styles.add(ParagraphStyle(name='SubM',fontName='Helvetica-Bold',fontSize=11.2,leading=14,
    textColor=NAVY,spaceBefore=8,spaceAfter=6))
styles.add(ParagraphStyle(name='SmallM',fontName='Times-Roman',fontSize=9,leading=12,
    textColor=GRAY,spaceAfter=7))
styles.add(ParagraphStyle(name='CellM',fontName='Helvetica',fontSize=8.5,leading=11))
styles.add(ParagraphStyle(name='CellH',fontName='Helvetica-Bold',fontSize=8.5,leading=11,textColor=colors.white))
styles.add(ParagraphStyle(name='TitleM',fontName='Helvetica-Bold',fontSize=28,leading=34,textColor=NAVY,alignment=TA_CENTER,spaceAfter=16))
styles.add(ParagraphStyle(name='CenterM',fontName='Times-Roman',fontSize=13,leading=19,alignment=TA_CENTER,spaceAfter=10))
styles.add(ParagraphStyle(name='CodeM',fontName='Courier',fontSize=9.3,leading=14,backColor=LIGHT,borderPadding=10,spaceAfter=10))
story=[]

def p(text,style='BodyM'):
    story.append(Paragraph(text,styles[style]))

def heading(text): p(text,'HeadM')
def sub(text): p(text,'SubM')
def small(text): p(text,'SmallM')
def page():
    while story and isinstance(story[-1],Spacer):
        story.pop()
    story.append(PageBreak())

def table(headers, rows, widths):
    data = [[Paragraph(str(x),styles['CellH']) for x in headers]]
    data += [[Paragraph(str(x),styles['CellM']) for x in row] for row in rows]
    t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),NAVY),('VALIGN',(0,0),(-1,-1),'TOP'),
        ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
        ('TOPPADDING',(0,0),(-1,-1),4.5),('BOTTOMPADDING',(0,0),(-1,-1),4.5),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,LIGHT]),
        ('LINEBELOW',(0,-1),(-1,-1),.6,colors.HexColor('#bac9d4'))]))
    story.extend([t,Spacer(1,8)])

def figure(name,width):
    from reportlab.lib.utils import ImageReader
    image = ImageReader(str(F/name)); w,h=image.getSize()
    story.append(Image(str(F/name),width=width,height=width*h/w))
    story.append(Spacer(1,5))

