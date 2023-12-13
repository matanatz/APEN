
import plotly.graph_objs as go
import numpy as np
from scipy.special import softmax

def get_scene_traces(pnts,outputs,normals,cap,layers,color="green",animation=False,k=0,box_predictions=True,seg_map=None):
        traces = []
        R = 16
        for iter in range(0,layers - 1 if cap != 'gt' else 1):
            if iter % R == 0:
                traces += [go.Mesh3d(
                            x=pnts[0,:,0].detach().cpu().numpy().tolist() + outputs['{0}_predicted_seeds'.format(iter//R)].sum(1)[0,:,0].detach().cpu().numpy().tolist() + (pnts[0,:,0] + 0.001).detach().cpu().numpy().tolist(),
                            y=pnts[0,:,1].detach().cpu().numpy().tolist() + outputs['{0}_predicted_seeds'.format(iter//R)].sum(1)[0,:,1].detach().cpu().numpy().tolist() + (pnts[0,:,1] + 0.001).detach().cpu().numpy().tolist(),
                            z=pnts[0,:,2].detach().cpu().numpy().tolist() + outputs['{0}_predicted_seeds'.format(iter//R)].sum(1)[0,:,2].detach().cpu().numpy().tolist() + (pnts[0,:,2] + 0.001).detach().cpu().numpy().tolist(),
                            i=list(range(pnts.shape[1])),
                            j=list(range(pnts.shape[1],2*pnts.shape[1])),
                            k=list(range(2*pnts.shape[1],3*pnts.shape[1])),
                            name="{0}-{1}_predicated_seeds".format(cap,iter),
                            showlegend=True,
                            visible="legendonly")]#iter//R + 1  < (layers-1)//R
                    #     )] + [go.Mesh3d(
                    #         x=pnts[0,:,0].detach().cpu().numpy().tolist() + outputs['{0}_predicated_seeds_other'.format(iter//R)].sum(1)[0,:,0].detach().cpu().numpy().tolist() + (pnts[0,:,0] + 0.001).detach().cpu().numpy().tolist(),
                    #         y=pnts[0,:,1].detach().cpu().numpy().tolist() + outputs['{0}_predicated_seeds_other'.format(iter//R)].sum(1)[0,:,1].detach().cpu().numpy().tolist() + (pnts[0,:,1] + 0.001).detach().cpu().numpy().tolist(),
                    #         z=pnts[0,:,2].detach().cpu().numpy().tolist() + outputs['{0}_predicated_seeds_other'.format(iter//R)].sum(1)[0,:,2].detach().cpu().numpy().tolist() + (pnts[0,:,2] + 0.001).detach().cpu().numpy().tolist(),
                    # #         colorbar_title='z',
                    # #         colorscale=[[0, 'gold'],
                    # #                     [0.5, 'mediumturquoise'],
                    # #                     [1, 'magenta']],
                    #         # Intensity of each vertex, which will be interpolated and color-coded
                    #         #intensity=[0, 0.33, 0.66, 1],
                    #         # i, j and k give the vertices of triangles
                    #         # here we represent the 4 triangles of the tetrahedron surface
                    #         i=list(range(pnts.shape[1])),
                    #         j=list(range(pnts.shape[1],2*pnts.shape[1])),
                    #         k=list(range(2*pnts.shape[1],3*pnts.shape[1])),
                    #         name="{0}-{1}_predicated_seeds_other".format(cap,iter),
                    #         showlegend=True,
                    #         visible="legendonly"
                    #     )]
            

            if (cap != "gt" and iter % R in [0,1,7,8,9,15]) or (cap == "gt" and iter % R == 0):
                traces += [go.Scatter3d(
                                        x=pnts.cpu()[0,:, 0],
                                        y=pnts.cpu()[0,:, 1],
                                        z=pnts.cpu()[0,:, 2],
                                        mode='markers',
                                        name="{0}-{1}_weights".format(cap,iter),
                                        visible="legendonly" , #iter < layers - 2 or cap == "gt"
                                        marker=dict(
                                            size=3,
                                            line=dict(
                                                width=2,
                                            ),
                                            opacity=0.9,
                                            showscale=False,
                                            color=outputs["{0}_weights".format(iter)].detach().cpu()[0].argmax(-1),
                                        ), text=[ np.array2string(np.sort(x[x>=0.05]) * 100,formatter={'float_kind':lambda x: "%.2f" % x},separator="<br>",max_line_width=10) for x in outputs["{0}_weights".format(iter)].detach().cpu().numpy()[0]])]
                traces += [
                     go.Cone(
                         x=pnts.cpu()[0,:, 0],
                        y=pnts.cpu()[0,:, 1],
                        z=pnts.cpu()[0,:, 2],
                         u=normals.cpu()[0,:,0],
                         v=normals.cpu()[0,:,1],
                         w=normals.cpu()[0,:,2],
                        sizemode="absolute",
                        sizeref=1,
                        anchor="tail",
                        name="{0}_normals".format(iter),
                        showscale=False,
                        showlegend=True,
                        visible="legendonly")

                ]
                
            if (cap != "gt" and iter % R in [0,1,7,8,9,15]) or (cap == "gt" and iter % R == 0):




                traces += [
                                        go.Scatter3d(
                                        x=outputs["{0}_predicted_candidates".format(iter)].detach().cpu()[0,:, 0],
                                        y=outputs["{0}_predicted_candidates".format(iter)].detach().cpu()[0,:, 1],
                                        z=outputs["{0}_predicted_candidates".format(iter)].detach().cpu()[0,:, 2],
                                        mode='markers',
                                        visible="legendonly" , #iter < layers - 2 or cap == "gt"
                                        name="{0}-{1}_predicted_candidates_p".format(cap,iter),
                                        legendgroup="{0}-{1}_predicted_candidates_p".format(cap,iter),
                                        marker=dict(
                                            size=7,
                                            line=dict(
                                                width=2,
                                            ),
                                            opacity=0.9,
                                            showscale=False,
                                            color=np.arange(outputs["{0}_predicted_candidates".format(iter)].shape[1])),
                                            text=outputs["{0}_predicted_pik".format(iter)].detach().cpu()[0,:])]
                if iter > 0 and cap != "gt" :
                    u, v = np.mgrid[0:np.pi:10j, 0:2*np.pi:10j]
                    x = (np.expand_dims(np.sqrt(outputs["{0}_predicted_var".format(iter)][0,:,0:1].detach().cpu().numpy()),-1) * np.expand_dims(np.sin(u) * np.cos(v),0))   + outputs["{0}_predicted_candidates".format(iter)].detach().cpu()[0,:, 0:1].unsqueeze(-1).numpy()
                    y = (np.expand_dims(np.sqrt(outputs["{0}_predicted_var".format(iter)][0,:,1:2].detach().cpu().numpy()),-1) *  np.expand_dims(np.sin(u) * np.sin(v),0) ) + outputs["{0}_predicted_candidates".format(iter)].detach().cpu()[0,:, 1:2].unsqueeze(-1).numpy()
                    z = (np.expand_dims(np.sqrt(outputs["{0}_predicted_var".format(iter)][0,:,2:3].detach().cpu().numpy()),-1) * np.expand_dims(np.cos(u),0)) + outputs["{0}_predicted_candidates".format(iter)].detach().cpu()[0,:, 2:3].unsqueeze(-1).numpy()
                    traces += [go.Surface(x=x[i], y=y[i], z=z[i],name="{0}-{1}_predicted_candidates".format(cap,iter),showscale=False,
                                    legendgroup="{0}-{1}_predicted_candidates".format(cap,iter),showlegend=i == outputs["{0}_candidate_mask".format(iter)][0].argmax().item(),visible="legendonly" if iter < layers - 1 or cap == "gt" else True) for i in range(x.shape[0]) if outputs["{0}_candidate_mask".format(iter)][0,i] == 1]
                                    # go.Mesh3d(
                                    # x=outputs["{0}_predicted_candidates".format(iter)][0,:,0].detach().cpu().numpy().tolist() + (outputs["{0}_predicted_candidates".format(iter)][0,:,0].detach().cpu() + np.sqrt(outputs["{0}_predicted_var".format(iter)][0,:,0].detach().cpu())).cpu().numpy().tolist() + (outputs["{0}_predicted_candidates".format(iter)][0,:,0] + 0.001).detach().cpu().numpy().tolist() +
                                    #   outputs["{0}_predicted_candidates".format(iter)][0,:,0].detach().cpu().numpy().tolist() + (outputs["{0}_predicted_candidates".format(iter)][0,:,0].detach().cpu() + 0*np.sqrt(outputs["{0}_predicted_var".format(iter)][0,:,0].detach().cpu())).cpu().numpy().tolist() + (outputs["{0}_predicted_candidates".format(iter)][0,:,0] + 0.001).detach().cpu().numpy().tolist() + 
                                    #   outputs["{0}_predicted_candidates".format(iter)][0,:,0].detach().cpu().numpy().tolist() + (outputs["{0}_predicted_candidates".format(iter)][0,:,0].detach().cpu() + 0*np.sqrt(outputs["{0}_predicted_var".format(iter)][0,:,0].detach().cpu())).cpu().numpy().tolist() + (outputs["{0}_predicted_candidates".format(iter)][0,:,0] + 0.001).detach().cpu().numpy().tolist(),
                                    # y=outputs["{0}_predicted_candidates".format(iter)][0,:,1].detach().cpu().numpy().tolist() + (outputs["{0}_predicted_candidates".format(iter)][0,:,1].detach().cpu() + 0*np.sqrt(outputs["{0}_predicted_var".format(iter)][0,:,1].detach().cpu())).cpu().numpy().tolist() + (outputs["{0}_predicted_candidates".format(iter)][0,:,1] + 0.001).detach().cpu().numpy().tolist() + 
                                    #   outputs["{0}_predicted_candidates".format(iter)][0,:,1].detach().cpu().numpy().tolist() + (outputs["{0}_predicted_candidates".format(iter)][0,:,1].detach().cpu() + np.sqrt(outputs["{0}_predicted_var".format(iter)][0,:,1].detach().cpu())).cpu().numpy().tolist() + (outputs["{0}_predicted_candidates".format(iter)][0,:,1] + 0.001).detach().cpu().numpy().tolist() + 
                                    #   outputs["{0}_predicted_candidates".format(iter)][0,:,1].detach().cpu().numpy().tolist() + (outputs["{0}_predicted_candidates".format(iter)][0,:,1].detach().cpu() + 0*np.sqrt(outputs["{0}_predicted_var".format(iter)][0,:,1].detach().cpu())).cpu().numpy().tolist() + (outputs["{0}_predicted_candidates".format(iter)][0,:,1] + 0.001).detach().cpu().numpy().tolist(),
                                    # z=outputs["{0}_predicted_candidates".format(iter)][0,:,2].detach().cpu().numpy().tolist() + (outputs["{0}_predicted_candidates".format(iter)][0,:,2].detach().cpu() + 0*np.sqrt(outputs["{0}_predicted_var".format(iter)][0,:,2].detach().cpu())).cpu().numpy().tolist() + (outputs["{0}_predicted_candidates".format(iter)][0,:,2] + 0.001).detach().cpu().numpy().tolist() + 
                                    # outputs["{0}_predicted_candidates".format(iter)][0,:,2].detach().cpu().numpy().tolist() + (outputs["{0}_predicted_candidates".format(iter)][0,:,2].detach().cpu() + 0*np.sqrt(outputs["{0}_predicted_var".format(iter)][0,:,2].detach().cpu())).cpu().numpy().tolist() + (outputs["{0}_predicted_candidates".format(iter)][0,:,2] + 0.001).detach().cpu().numpy().tolist() + 
                                    # outputs["{0}_predicted_candidates".format(iter)][0,:,2].detach().cpu().numpy().tolist() + (outputs["{0}_predicted_candidates".format(iter)][0,:,2].detach().cpu() + np.sqrt(outputs["{0}_predicted_var".format(iter)][0,:,2].detach().cpu())).cpu().numpy().tolist() + (outputs["{0}_predicted_candidates".format(iter)][0,:,2] + 0.001).detach().cpu().numpy().tolist(),
                    #         colorbar_title='z',
                    #         colorscale=[[0, 'gold'],
                    #                     [0.5, 'mediumturquoise'],
                    #                     [1, 'magenta']],
                            # Intensity of each vertex, which will be interpolated and color-coded
                            #intensity=[0, 0.33, 0.66, 1],
                            # i, j and k give the vertices of triangles
                            # here we represent the 4 triangles of the tetrahedron surface
                            # i=list(range(outputs["{0}_predicted_candidates".format(iter)].shape[1])) +  list(range(3*outputs["{0}_predicted_candidates".format(iter)].shape[1],4*outputs["{0}_predicted_candidates".format(iter)].shape[1])),
                            # j=list(range(outputs["{0}_predicted_candidates".format(iter)].shape[1],2*outputs["{0}_predicted_candidates".format(iter)].shape[1])) + list(range(4*outputs["{0}_predicted_candidates".format(iter)].shape[1],5*outputs["{0}_predicted_candidates".format(iter)].shape[1])),
                            # k=list(range(2*outputs["{0}_predicted_candidates".format(iter)].shape[1],3*outputs["{0}_predicted_candidates".format(iter)].shape[1])) + list(range(6*outputs["{0}_predicted_candidates".format(iter)].shape[1],7*outputs["{0}_predicted_candidates".format(iter)].shape[1])),
                            # name="{0}-{1}_predicted_candidates".format(cap,iter),
                            # legendgroup="{0}-{1}_predicted_candidates".format(cap,iter),
                            # showlegend=False,
                            # visible="legendonly"
                            #     )]
                                    # go.Scatter3d(
                                    # x=outputs["final_center"].detach().cpu()[0,:, 0],
                                    # y=outputs["final_center"].detach().cpu()[0,:, 1],
                                    # z=outputs["final_center"].detach().cpu()[0,:, 2],
                                    # mode='markers',
                                    # name="{0}-final_center".format(cap),
                                    # marker=dict(
                                    #     size=7,
                                    #     line=dict(
                                    #         width=2,
                                    #     ),
                                    #     opacity=0.9,
                                    #     showscale=True,
                                    #     color="blue",
                                    # ))]
        
        
        # AFTER FOR LOOP
        additions = [go.Scatter3d(
                                        x=pnts.cpu()[0,:, 0],
                                        y=pnts.cpu()[0,:, 1],
                                        z=pnts.cpu()[0,:, 2],
                                        mode='markers',
                                        name="part_seg_pred".format(cap,iter),
                                        visible="legendonly", #iter < layers - 2 or cap == "gt"
                                        marker=dict(
                                            size=3,
                                            line=dict(
                                                width=2,
                                            ),
                                            opacity=0.9,
                                            showscale=False,
                                            color=outputs["seed_part_logits"].detach().cpu()[0].argmax(-1),
                                        ), text=[ np.array2string(x.argmax(-1),
                                                                  formatter={'float_kind':lambda x: "%.2f" % x},
                                                                  separator="<br>",max_line_width=10) if seg_map is None else seg_map[x.argmax(-1)] for x in outputs["seed_part_logits"].detach().cpu().numpy()[0]])]
        
        additions += [
            # go.Mesh3d(
            #                 x=pnts[0,:,0].detach().cpu().numpy().tolist() + (pnts+0.1*normals)[0,:,0].detach().cpu().numpy().tolist() + (pnts[0,:,0] + 0.001).detach().cpu().numpy().tolist(),
            #                 y=pnts[0,:,1].detach().cpu().numpy().tolist() + (pnts+0.1*normals)[0,:,1].detach().cpu().numpy().tolist() + (pnts[0,:,1] + 0.001).detach().cpu().numpy().tolist(),
            #                 z=pnts[0,:,2].detach().cpu().numpy().tolist() + (pnts+0.1*normals)[0,:,2].detach().cpu().numpy().tolist() + (pnts[0,:,2] + 0.001).detach().cpu().numpy().tolist(),
            #         #         colorbar_title='z',
            #         #         colorscale=[[0, 'gold'],
            #         #                     [0.5, 'mediumturquoise'],
            #         #                     [1, 'magenta']],
            #                 # Intensity of each vertex, which will be interpolated and color-coded
            #                 #intensity=[0, 0.33, 0.66, 1],
            #                 # i, j and k give the vertices of triangles
            #                 # here we represent the 4 triangles of the tetrahedron surface
            #                 i=list(range(pnts.shape[1])),
            #                 j=list(range(pnts.shape[1],2*pnts.shape[1])),
            #                 k=list(range(2*pnts.shape[1],3*pnts.shape[1])),
            #                 name="normals",
            #                 showlegend=True,
            #                 visible="legendonly"
            #             ),
             go.Scatter3d(
                                        x=pnts.cpu()[0,:, 0],
                                        y=pnts.cpu()[0,:, 1],
                                        z=pnts.cpu()[0,:, 2],
                                        mode='markers',
                                        name="init_points".format(cap,iter),
                                        visible="legendonly", #iter < layers - 2 or cap == "gt"
                                        marker=dict(
                                            size=3,
                                            line=dict(
                                                width=2,
                                            ),
                                            opacity=0.9,
                                            showscale=False,
                                        )), 
            go.Scatter3d(
                                        x=pnts.cpu()[0,:, 0],
                                        y=pnts.cpu()[0,:, 1],
                                        z=pnts.cpu()[0,:, 2],
                                        mode='markers',
                                        name="init_weights".format(cap,iter),
                                        visible="legendonly", #iter < layers - 2 or cap == "gt"
                                        marker=dict(
                                            size=3,
                                            line=dict(
                                                width=2,
                                            ),
                                            opacity=0.9,
                                            showscale=False,
                                            color=outputs["init_weights"].detach().cpu()[0].argmax(-1),
                                        ), text=[ np.array2string(np.sort(x[x>=0.0]) * 100,formatter={'float_kind':lambda x: "%.2f" % x},separator="<br>",max_line_width=10) for x in outputs["init_weights"].detach().cpu().numpy()[0]]),
            go.Scatter3d(
                                x=outputs["{0}_final_centers".format(layers//16)].detach().cpu()[0,:, 0],
                                y=outputs["{0}_final_centers".format(layers//16)].detach().cpu()[0,:, 1],
                                z=outputs["{0}_final_centers".format(layers//16)].detach().cpu()[0,:, 2],
                                mode='markers',
                                visible="legendonly",
                                name="{0}-final_center".format(cap),
                                marker=dict(
                                    size=7,
                                    line=dict(
                                        width=2,
                                    ),
                                    opacity=0.9,
                                    showscale=False,
                                    color="blue",
                                )),
                        go.Scatter3d(
                                x=outputs["init_candidates"].detach().cpu()[0,:, 0],
                                y=outputs["init_candidates"].detach().cpu()[0,:, 1],
                                z=outputs["init_candidates"].detach().cpu()[0,:, 2],
                                mode='markers',
                                name="{0}_init_candidates".format(cap),
                                visible="legendonly",
                                marker=dict(
                                    size=7,
                                    line=dict(
                                        width=2,
                                    ),
                                    opacity=0.9,
                                    showscale=False,
                                    color=np.arange(outputs["init_candidates"].shape[1]),
                                ))]
                        
        
        
        # box preidctions
        if box_predictions and False:
            mask = outputs['{0}_final_candidate_mask'.format(layers//16)][0].bool()
            center = outputs["{0}_final_centers".format(layers//16)][0].masked_select(mask.unsqueeze(-1).tile(1,3)).view(-1,3)

            pred_size_class = outputs['size_scores'][0].argmax(-1)
            pred_size_residual = utils.vector_gather(outputs['size_residuals'][0],pred_size_class).masked_select(mask.unsqueeze(-1).tile(1,3)).view(-1,3)
            box_size = []
            for i in range(pred_size_residual.shape[0]):
                box_size.append(torch.tensor(dataset_config.class2size(int(pred_size_class[i].detach().cpu().numpy()), pred_size_residual[i].detach().cpu().numpy())).unsqueeze(0))

            box_size = torch.cat(box_size,dim=0).to(center)
            R = outputs['heading_rotation_matrix'][0].masked_select(mask.unsqueeze(-1).unsqueeze(-1).tile(1,3,3)).view(-1,3,3).transpose(1,2)
            opts = torch.tensor([[1,1,1],
                                                    [1,1,-1],
                                                    [1,-1,1],
                                                    [1,-1,-1],
                                                    [-1,1,1],
                                                    [-1,1,-1],
                                                    [-1,-1,1],
                                                    [-1,-1,-1]]).to(box_size)
            corners = (opts.unsqueeze(0).tile(box_size.shape[0],1,1) * box_size.unsqueeze(1)).transpose(1,2)
            corners_3d = (torch.matmul(R,corners) + center.unsqueeze(-1)).transpose(1,2)
            # points_3d = corners_3d.reshape(-1,3)
            # all_i = []
            # all_j = []
            # all_k = []
            # for j in range(0,corners_3d.shape[0]):
            #     all_i += (np.array([7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2]) + j*8).tolist()
            #     all_j += (np.array([3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3]) + j*8).tolist()
            #     all_k += (np.array([0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6]) + j*8).tolist()

            # simplices = np.array(list(zip(all_i,all_j,all_k)))
            # tri_vertices=list(map(lambda index: points_3d[index], simplices))
            # lists_coord=[[[T[k%3][c] for k in range(4)]+[ None] for T in tri_vertices] for c in range(3)]
            # Xe, Ye, Ze=[functools.reduce(lambda x,y: x+y, lists_coord[k]) for k in range(3)]
            edges = torch.tensor([[0,1],[1,3],[3,2],[2,0],[0,4],[4,6],[6,2],[2,6],[6,7],[7,3],[3,7],[7,5],[5,4],[4,5],[5,1]]).to(corners_3d).long()
            edges = edges.unsqueeze(0).tile(corners_3d.shape[0],1,1)
            corners_3d = corners_3d.unsqueeze(1).tile(1,edges.shape[1],1,1)
            if animation:
                additions = [ go.Scatter3d(
                                    x=outputs["{0}_final_centers".format(layers//16)].detach().cpu()[0,:, 0],
                                    y=outputs["{0}_final_centers".format(layers//16)].detach().cpu()[0,:, 1],
                                    z=outputs["{0}_final_centers".format(layers//16)].detach().cpu()[0,:, 2],
                                    mode='markers',
                                    visible=False,
                                    name="{0}-final_center".format(cap),
                                    marker=dict(
                                        size=7,
                                        line=dict(
                                            width=2,
                                        ),
                                        opacity=0.9,
                                        showscale=False,
                                        color="blue",
                                    )),
                                    go.Scatter3d(
                                                x=pnts.cpu()[0,:, 0],
                                                y=pnts.cpu()[0,:, 1],
                                                z=pnts.cpu()[0,:, 2],
                                                mode='markers',
                                                name="{0}-{1}_weights".format(cap,iter),
                                                visible="legendonly" if iter < layers - 2 or cap == "gt"  else True,
                                                marker=dict(
                                                    size=3,
                                                    line=dict(
                                                        width=2,
                                                    ),
                                                    opacity=0.9,
                                                    showscale=False,
                                                    #color=outputs["{0}_weights".format(iter)].detach().cpu()[0].argmax(-1),
                                                ), text=[ np.array2string(np.sort(x[x>=0.05]) * 100,formatter={'float_kind':lambda x: "%.2f" % x},separator="<br>",max_line_width=10) for x in outputs["{0}_weights".format(iter)].detach().cpu().numpy()[0]])]
            order = center[:,0].argsort()
            for i,(c,e,color) in enumerate(zip(corners_3d[order],edges[order],outputs['sem_cls_scores'][0].masked_select(mask.unsqueeze(-1).tile(1,outputs['sem_cls_scores'][0].shape[-1])).view(-1,outputs['sem_cls_scores'][0].shape[-1]).argmax(-1).detach().cpu()[order])):
                lxyz = utils.vector_gather(c,e).view(-1,3).cpu().detach().numpy()
                #print (i,color.item())
                lines=go.Scatter3d(x=lxyz[:,0],y=lxyz[:,1],z=lxyz[:,2],mode='lines',line=go.Line(color=px.colors.qualitative.Plotly[color.item()], width=3.5),name='box_{0}_{1}'.format(k,cap),
                    showlegend=i==0,visible=True,legendgroup="box_{0}".format(cap))
                additions += [lines]
        if animation:
            return additions
        else:
            return traces + additions
    
