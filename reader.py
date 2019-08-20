import csv

class Reader():
    def read(self, filename, interactive=False, prediction=False, price=False):
        with open(filename, newline='\n') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            x_result = []
            y_result = []
            for row in reader:
                if prediction:
                    x_result.append({"name": row[0], "age": row[1], "club": row[2], "position": row[3], "pred_pts": float(row[4])})
                    y_result.append(0)
                    continue
                if price:
                    x_result.append({"name": row[0], "price": float(row[1]), "club": row[2], "position": row[3]})
                    y_result.append(0)
                    continue
                x_result.append({"name": row[0], "position": row[1], "age": row[2], "club": row[3]})
                if interactive:
                    y_result.append(0)
                else:
                    y_result.append(float(row[4].replace(",", ".")))
            return x_result, y_result
