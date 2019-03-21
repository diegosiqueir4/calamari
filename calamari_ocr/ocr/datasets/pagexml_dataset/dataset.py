import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from lxml import etree
from skimage.draw import polygon
from typing import List

from calamari_ocr.ocr.datasets import DataSet, DataSetMode
from calamari_ocr.utils import split_all_ext

from multiprocessing import Queue, Process, Manager
from queue import Full
import random


class PageXMLReader:
    def __init__(self, mode, non_existing_as_empty, text_index, skip_invalid):
        self.mode = mode
        self.text_index = text_index
        self.skip_invalid = skip_invalid
        self._non_existing_as_empty = non_existing_as_empty

    def read_page_xml(self, img, xml, skipcommented=True):
        if not os.path.exists(xml):
            if self._non_existing_as_empty:
                return None
            else:
                raise Exception("File '{}' does not exist.".format(xml))

        root = etree.parse(xml).getroot()

        if self.mode == DataSetMode.TRAIN_ON_THE_FLY or self.mode == DataSetMode.TRAIN or self.mode == DataSetMode.EVAL:
            return root, self._samples_gt_from_book(root, img, skipcommented)
        else:
            return root, self._samples_from_book(root, img)

    def _samples_gt_from_book(self, root, img,
                              skipcommented=True):
        ns = {"ns": root.nsmap[None]}
        imgfile = root.xpath('//ns:Page',
                             namespaces=ns)[0].attrib["imageFilename"]
        if (self.mode == DataSetMode.TRAIN_ON_THE_FLY or self.mode == DataSetMode.TRAIN) and not img.endswith(imgfile):
            raise Exception("Mapping of image file to xml file invalid: {} vs {}".format(img, imgfile))

        img_w = int(root.xpath('//ns:Page',
                               namespaces=ns)[0].attrib["imageWidth"])
        tequivs = root.xpath('//ns:TextEquiv[@index="{}"]'.format(self.text_index),
                             namespaces=ns)
        for l in tequivs:
            parat = l.getparent().attrib
            if skipcommented and "comments" in parat and parat["comments"]:
                continue

            text = l.xpath('./ns:Unicode', namespaces=ns).pop().text
            if not text:
                if self.skip_invalid:
                    continue
                elif self._non_existing_as_empty:
                    text = ""
                else:
                    raise Exception("Empty text field")

            yield {
                'ns': ns,
                "rtype": l.xpath('../../@type', namespaces=ns).pop(),
                'xml_element': l,
                "imgfile": img,
                "id": l.xpath('../@id', namespaces=ns).pop(),
                "text": text,
                "coords": l.xpath('../ns:Coords/@points', namespaces=ns).pop(),
                "img_width": img_w
            }

    def _samples_from_book(self, root, img):
        ns = {"ns": root.nsmap[None]}
        imgfile = root.xpath('//ns:Page',
                             namespaces=ns)[0].attrib["imageFilename"]
        if not img.endswith(imgfile):
            raise Exception("Mapping of image file to xml file invalid: {} vs {}".format(img, imgfile))

        img_w = int(root.xpath('//ns:Page',
                               namespaces=ns)[0].attrib["imageWidth"])
        for l in root.xpath('//ns:TextLine', namespaces=ns):
            yield {
                 'ns': ns,
                "rtype": l.xpath('../@type', namespaces=ns).pop(),
                'xml_element': l,
                "imgfile": img,
                "id": l.xpath('./@id', namespaces=ns).pop(),
                "coords": l.xpath('./ns:Coords/@points',
                                  namespaces=ns).pop(),
                "img_width": img_w,
                "text": None,
            }


class PageXMLLoaderProcess(Process):
    def __init__(self, input_files, output_queue: Queue, reader: PageXMLReader, name=-1):
        super().__init__()
        self.input_files = input_files
        self.output_queue = output_queue
        self.name = "{}".format(name)
        self.reader = reader

    def _handle(self):
        img_file, page_xml = random.choice(self.input_files)

        img = np.array(Image.open(img_file))
        ly, lx = img.shape
        for sample in self.reader.read_page_xml(img_file, page_xml)[1]:
            text = sample["text"]
            line_img = PageXMLDataset.cutout(img, sample['coords'], lx / sample['img_width'])
            self.output_queue.put((line_img, text))

    def run(self):
        random.seed()
        np.random.seed()
        try:
            while True:
                self._handle()
        except (EOFError, BrokenPipeError, ConnectionResetError):
            # queue closed, stop the process
            return


class PageXMLDataset(DataSet):

    def __init__(self,
                 mode: DataSetMode,
                 files,
                 xmlfiles: List[str] = None,
                 skip_invalid=False,
                 remove_invalid=True,
                 non_existing_as_empty=False,
                 args: dict = None,
                 ):

        """ Create a dataset from a Path as String

        Parameters
         ----------
        files : [], required
            image files
        skip_invalid : bool, optional
            skip invalid files
        remove_invalid : bool, optional
            remove invalid files
        """

        super().__init__(
            mode,
            skip_invalid, remove_invalid,
        )

        if xmlfiles is None:
            xmlfiles = []

        if args is None:
            args = []

        self.text_index = args.get('text_index', 0)

        self._non_existing_as_empty = non_existing_as_empty
        if len(xmlfiles) == 0:
            xmlfiles = [split_all_ext(p)[0] + ".xml" for p in files]

        if len(files) == 0:
            files = [None] * len(xmlfiles)

        self.files = files
        self.xmlfiles = xmlfiles
        self.reader = PageXMLReader(mode, non_existing_as_empty, self.text_index, skip_invalid)

        # self.pages = [self.read_page_xml(img, xml) for img, xml in zip(files, xmlfiles)]
        self.pages = []
        for img, xml in zip(files, xmlfiles):
            root, samples = self.reader.read_page_xml(img, xml)
            self.pages.append(root)
            for sample in samples:
                self.add_sample(sample)

        # during training, to support loading on the fly, create runners that select random files and write all
        # their content into a queue, that is read as 'single_sample'
        self.manager = Manager()
        if self.mode == DataSetMode.TRAIN_ON_THE_FLY:
            self.queue = self.manager.Queue(100)
            runners = [PageXMLLoaderProcess(list(zip(files, xmlfiles)), self.queue, self.reader) for _ in range(args.get('processes', 1))]
            for r in runners:
                r.start()

            # pages and elements can not be stored in a training dataset, because they can not be serialized
            self.pages = []
            for s in self.samples():
                s['xml_element'] = None
        else:
            self.queue = None

    def __getstate__(self):
        # pickle only relevant information to load samples, drop all irrelevant
        return self.mode, self.text_index, self._non_existing_as_empty, self.files, self.xmlfiles, self.reader, self.queue

    def __setstate__(self, state):
        self.mode, self.text_index, self._non_existing_as_empty, self.files, self.xmlfiles, self.reader, self.queue = state

    @staticmethod
    def cutout(pageimg, coordstring, scale=1, rect=False):
        coords = [p.split(",") for p in coordstring.split()]
        coords = np.array([(int(scale * int(c[1])), int(scale * int(c[0])))
                           for c in coords])
        if rect:
            return pageimg[min(c[0] for c in coords):max(c[0] for c in coords),
                   min(c[1] for c in coords):max(c[1] for c in coords)]
        rr, cc = polygon(coords[:, 0], coords[:, 1], pageimg.shape)
        offset = (min([x[0] for x in coords]), min([x[1] for x in coords]))
        box = np.ones(
            (max([x[0] for x in coords]) - offset[0],
             max([x[1] for x in coords]) - offset[1]),
            dtype=pageimg.dtype) * 255
        box[rr - offset[0], cc - offset[1]] = pageimg[rr, cc]
        return box

    def _load_sample(self, sample, text_only):
        image_path = sample["imgfile"]
        text = sample["text"]
        img = None

        if text_only:
            return img, text
        elif self.mode == DataSetMode.TRAIN.TRAIN_ON_THE_FLY:
            return self.queue.get()
        elif self.mode == DataSetMode.PREDICT or self.mode == DataSetMode.TRAIN:
            img = np.array(Image.open(image_path))

            ly, lx = img.shape

            img = PageXMLDataset.cutout(img, sample['coords'], lx / sample['img_width'])

            # add padding as required from normal files
            # img = np.pad(img, ((3, 3), (0, 0)), mode='constant', constant_values=img.max())

        return img, text

    def store_text(self, sentence, sample, output_dir, extension):
        ns = sample['ns']
        line = sample['xml_element']
        textequivxml = line.find('./ns:TextEquiv[@index="{}"]'.format(self.text_index),
                                    namespaces=ns)
        if textequivxml is None:
            textequivxml = etree.SubElement(line, "TextEquiv", attrib={"index": str(self.text_index)})

        u_xml = textequivxml.find('./ns:Unicode', namespaces=ns)
        if u_xml is None:
            u_xml = etree.SubElement(textequivxml, "Unicode")

        u_xml.text = sentence

    def store(self):
        for xml, page in tqdm(zip(self.xmlfiles, self.pages), desc="Writing PageXML files", total=len(self.xmlfiles)):
            with open(split_all_ext(xml)[0] + ".pred.xml", 'w') as f:
                f.write(etree.tounicode(page.getroottree()))
