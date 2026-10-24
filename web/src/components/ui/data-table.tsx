"use client";

import * as React from "react";
import * as DropdownMenu from "@radix-ui/react-dropdown-menu";
import {
  type ColumnDef,
  type SortingState,
  type VisibilityState,
  flexRender,
  getCoreRowModel,
  getFilteredRowModel,
  getSortedRowModel,
  getPaginationRowModel,
  useReactTable,
} from "@tanstack/react-table";
import { ArrowDown, ArrowUp, ChevronsUpDown, Columns3, Download, Search } from "lucide-react";
import { cn, downloadCsv } from "@/lib/utils";
import { Button, EmptyState, Input, Skeleton } from "./primitives";

export type { ColumnDef };

/**
 * Dense data table: sorting, text filter, column visibility, CSV export (of the filtered rows), pagination.
 * Numeric columns are right-aligned with tabular figures (set `meta: { numeric: true }` on the column).
 */
export function DataTable<T extends object>({
  columns,
  data,
  loading,
  csvName,
  onRowClick,
  emptyTitle = "Nothing here yet",
  emptyBody = "",
  toolbar,
  pageSize = 25,
  searchPlaceholder = "Filter…",
  initialSorting = [],
  rowLabel,
}: {
  columns: ColumnDef<T, unknown>[];
  data: T[] | undefined;
  loading?: boolean;
  csvName?: string;
  onRowClick?: (row: T) => void;
  emptyTitle?: string;
  emptyBody?: React.ReactNode;
  toolbar?: React.ReactNode;
  pageSize?: number;
  searchPlaceholder?: string;
  initialSorting?: SortingState;
  rowLabel?: (row: T) => string;
}) {
  const [sorting, setSorting] = React.useState<SortingState>(initialSorting);
  const [filter, setFilter] = React.useState("");
  const [visibility, setVisibility] = React.useState<VisibilityState>({});
  // TanStack Table returns non-memoizable functions; the React Compiler skips this component, which is fine here.
  // eslint-disable-next-line react-hooks/incompatible-library
  const table = useReactTable({
    data: data ?? [],
    columns,
    state: { sorting, globalFilter: filter, columnVisibility: visibility },
    onSortingChange: setSorting,
    onGlobalFilterChange: setFilter,
    onColumnVisibilityChange: setVisibility,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    initialState: { pagination: { pageSize } },
  });
  const rows = table.getRowModel().rows;
  const exportCsv = () =>
    downloadCsv(
      `${csvName ?? "export"}.csv`,
      table.getFilteredRowModel().rows.map((r) => r.original as Record<string, unknown>),
    );

  return (
    <div className="flex flex-col">
      <div className="flex flex-wrap items-center gap-2 border-b border-border px-3 py-2">
        <div className="relative min-w-40 flex-1 sm:max-w-64">
          <Search aria-hidden className="pointer-events-none absolute left-2 top-1/2 size-3.5 -translate-y-1/2 text-fg-muted" />
          <Input
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            placeholder={searchPlaceholder}
            aria-label="Filter rows"
            className="h-7 pl-7 text-xs"
          />
        </div>
        {toolbar}
        <div className="ml-auto flex items-center gap-1.5">
          <DropdownMenu.Root>
            <DropdownMenu.Trigger asChild>
              <Button variant="ghost" size="sm" aria-label="Choose columns">
                <Columns3 /> Columns
              </Button>
            </DropdownMenu.Trigger>
            <DropdownMenu.Portal>
              <DropdownMenu.Content align="end" sideOffset={4} className="z-50 min-w-44 rounded-md border border-border bg-surface-2 p-1 shadow-card">
                {table
                  .getAllLeafColumns()
                  .filter((c) => c.getCanHide())
                  .map((c) => (
                    <DropdownMenu.CheckboxItem
                      key={c.id}
                      checked={c.getIsVisible()}
                      onCheckedChange={(v) => c.toggleVisibility(Boolean(v))}
                      onSelect={(e) => e.preventDefault()}
                      className="flex cursor-pointer items-center gap-2 rounded-sm px-2 py-1 text-xs text-fg outline-none data-[highlighted]:bg-surface-3"
                    >
                      <span className={cn("size-3 rounded-sm border border-border-strong", c.getIsVisible() && "border-accent bg-accent")} />
                      {typeof c.columnDef.header === "string" ? c.columnDef.header : c.id}
                    </DropdownMenu.CheckboxItem>
                  ))}
              </DropdownMenu.Content>
            </DropdownMenu.Portal>
          </DropdownMenu.Root>
          {csvName ? (
            <Button variant="ghost" size="sm" onClick={exportCsv} disabled={!data?.length}>
              <Download /> CSV
            </Button>
          ) : null}
        </div>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full border-collapse text-[13px]">
          <thead>
            {table.getHeaderGroups().map((hg) => (
              <tr key={hg.id} className="border-b border-border">
                {hg.headers.map((h) => {
                  const numeric = (h.column.columnDef.meta as { numeric?: boolean } | undefined)?.numeric;
                  const sorted = h.column.getIsSorted();
                  return (
                    <th
                      key={h.id}
                      scope="col"
                      aria-sort={sorted === "asc" ? "ascending" : sorted === "desc" ? "descending" : "none"}
                      className={cn(
                        "whitespace-nowrap px-3 py-2 text-[11px] font-medium uppercase tracking-wide text-fg-muted",
                        numeric ? "text-right" : "text-left",
                      )}
                    >
                      {h.isPlaceholder ? null : h.column.getCanSort() ? (
                        <button
                          type="button"
                          onClick={h.column.getToggleSortingHandler()}
                          className={cn("inline-flex items-center gap-1 hover:text-fg", numeric && "flex-row-reverse")}
                        >
                          {flexRender(h.column.columnDef.header, h.getContext())}
                          {sorted === "asc" ? (
                            <ArrowUp className="size-3" />
                          ) : sorted === "desc" ? (
                            <ArrowDown className="size-3" />
                          ) : (
                            <ChevronsUpDown className="size-3 opacity-40" />
                          )}
                        </button>
                      ) : (
                        flexRender(h.column.columnDef.header, h.getContext())
                      )}
                    </th>
                  );
                })}
              </tr>
            ))}
          </thead>
          <tbody>
            {loading
              ? Array.from({ length: 6 }).map((_, i) => (
                  <tr key={i} className="border-b border-border/60">
                    {table.getVisibleLeafColumns().map((c) => (
                      <td key={c.id} className="px-3 py-2.5">
                        <Skeleton className="h-3.5" />
                      </td>
                    ))}
                  </tr>
                ))
              : rows.map((r) => (
                  <tr
                    key={r.id}
                    onClick={onRowClick ? () => onRowClick(r.original) : undefined}
                    onKeyDown={
                      onRowClick
                        ? (e) => {
                            if (e.key === "Enter" || e.key === " ") {
                              e.preventDefault();
                              onRowClick(r.original);
                            }
                          }
                        : undefined
                    }
                    tabIndex={onRowClick ? 0 : undefined}
                    aria-label={onRowClick && rowLabel ? rowLabel(r.original) : undefined}
                    className={cn(
                      "border-b border-border/60 transition-colors duration-150",
                      onRowClick && "cursor-pointer hover:bg-surface-2 focus-visible:bg-surface-2",
                    )}
                  >
                    {r.getVisibleCells().map((cell) => {
                      const numeric = (cell.column.columnDef.meta as { numeric?: boolean } | undefined)?.numeric;
                      return (
                        <td key={cell.id} className={cn("whitespace-nowrap px-3 py-2 text-fg", numeric && "num text-right font-mono text-xs")}>
                          {flexRender(cell.column.columnDef.cell, cell.getContext())}
                        </td>
                      );
                    })}
                  </tr>
                ))}
          </tbody>
        </table>
      </div>
      {!loading && rows.length === 0 ? <EmptyState title={emptyTitle} body={emptyBody} /> : null}
      {table.getPageCount() > 1 ? (
        <div className="flex items-center justify-between border-t border-border px-3 py-2 text-xs text-fg-muted">
          <span className="num">
            {table.getFilteredRowModel().rows.length.toLocaleString()} rows · page {table.getState().pagination.pageIndex + 1} of{" "}
            {table.getPageCount()}
          </span>
          <div className="flex gap-1">
            <Button size="sm" variant="ghost" onClick={() => table.previousPage()} disabled={!table.getCanPreviousPage()}>
              Previous
            </Button>
            <Button size="sm" variant="ghost" onClick={() => table.nextPage()} disabled={!table.getCanNextPage()}>
              Next
            </Button>
          </div>
        </div>
      ) : null}
    </div>
  );
}
