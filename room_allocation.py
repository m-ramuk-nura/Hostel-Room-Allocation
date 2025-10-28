import os
import pandas as pd
import math

class RoomAllocator:
    def __init__(self, folder_path='clusters', ranking_col='Health_Condition_Rank', room_col='room_number', room_size=4):
        self.folder_path = folder_path
        self.ranking_col = ranking_col
        self.room_col = room_col
        self.room_size = room_size
        self.dataframes = []
        self.csv_files = []
        self.room_counter = 1

    def load_data(self):
        self.csv_files = sorted([f for f in os.listdir(self.folder_path) if f.endswith('.csv')])
        self.dataframes = [
            pd.read_csv(os.path.join(self.folder_path, f)).assign(_file=f)
            for f in self.csv_files
        ]

        for df in self.dataframes:
            df.columns = df.columns.str.strip()
            if self.room_col not in df.columns:
                df[self.room_col] = pd.NA

    def fill_room_in_sheet(self, df, start_pos, already_allocated_indices):
        allocated = already_allocated_indices.copy()
        n = len(df)

        for i in range(start_pos + 1, n):
            if pd.isna(df.loc[i, self.room_col]) and i not in allocated:
                allocated.append(i)
                if len(allocated) == self.room_size:
                    return allocated
        for i in range(0, start_pos):
            if pd.isna(df.loc[i, self.room_col]) and i not in allocated:
                allocated.append(i)
                if len(allocated) == self.room_size:
                    return allocated
        return allocated

    def fill_remaining_from_next_sheets(self, current_sheet_idx, allocated_indices_global):
        allocated = allocated_indices_global.copy()
        for idx_sheet in range(current_sheet_idx + 1, len(self.dataframes)):
            df_next = self.dataframes[idx_sheet]
            for i in range(len(df_next)):
                if pd.isna(df_next.loc[i, self.room_col]) and (idx_sheet, i) not in allocated:
                    allocated.append((idx_sheet, i))
                    if len(allocated) == self.room_size:
                        return allocated
        return allocated

    def allocate_rooms(self):
        all_ranks = set()
        for df in self.dataframes:
            all_ranks.update(df[self.ranking_col].dropna().unique())
        all_ranks.discard(-1)
        all_ranks = sorted(all_ranks)

        for rank in all_ranks:
            for sheet_idx, df in enumerate(self.dataframes):
                pos = 0
                while pos < len(df):
                    while pos < len(df) and not (pd.isna(df.loc[pos, self.room_col]) and int(df.loc[pos, self.ranking_col]) == rank):
                        pos += 1
                    if pos == len(df):
                        break

                    allocated_local = [pos]
                    allocated_local = self.fill_room_in_sheet(df, pos, allocated_local)

                    if len(allocated_local) < self.room_size:
                        allocated_local = self.fill_remaining_from_next_sheets(
                            sheet_idx, [(sheet_idx, idx) for idx in allocated_local]
                        )

                    for alloc in allocated_local:
                        if isinstance(alloc, tuple):
                            ds_idx, row_idx = alloc
                            self.dataframes[ds_idx].loc[row_idx, self.room_col] = self.room_counter
                        else:
                            df.loc[alloc, self.room_col] = self.room_counter

                    self.room_counter += 1
                    pos += 1

        self.allocate_remaining()

    def allocate_remaining(self):
        for sheet_idx, df in enumerate(self.dataframes):
            pos = 0
            while pos < len(df):
                if pd.isna(df.loc[pos, self.room_col]):
                    allocated_local = [pos]
                    allocated_local = self.fill_room_in_sheet(df, pos, allocated_local)
                    if len(allocated_local) < self.room_size:
                        allocated_local = self.fill_remaining_from_next_sheets(
                            sheet_idx, [(sheet_idx, idx) for idx in allocated_local]
                        )

                    for alloc in allocated_local:
                        if isinstance(alloc, tuple):
                            ds_idx, row_idx = alloc
                            self.dataframes[ds_idx].loc[row_idx, self.room_col] = self.room_counter
                        else:
                            df.loc[alloc, self.room_col] = self.room_counter

                    self.room_counter += 1
                    pos += 1
                else:
                    pos += 1

    def save_results(self):
        for df in self.dataframes:
            file_name = df['_file'].iloc[0]
            df.drop(columns=['_file'], inplace=True)
            df.to_csv(os.path.join(self.folder_path, file_name), index=False)

        combined = pd.concat(self.dataframes, ignore_index=True)
        combined.to_csv(os.path.join('combined_output.csv'), index=False)


    def run(self):
        self.load_data()
        self.allocate_rooms()
        self.save_results()
