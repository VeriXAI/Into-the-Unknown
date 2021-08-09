class CsvStorage:
    def __init__(self, rows, row_header):
        self.rows = rows
        self.row_header = row_header

    @staticmethod
    def row_header(self):
        return self.row_header

    def write(self, writer):
        for row in self.rows:
            writer.writerow(row)
